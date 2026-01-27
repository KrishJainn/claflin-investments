"""
Indicator Calculator module.

Batch calculates technical indicators using 'ta' library.
Supports efficient calculation of indicators in a single pass.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
import logging

try:
    import ta
    from ta import momentum, trend, volatility, volume
except ImportError:
    ta = None
    logging.warning("'ta' library not installed. Run: pip install ta")

from .universe import IndicatorUniverse, IndicatorDefinition
from .normalizer import IndicatorNormalizer, NormalizationConfig

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """
    Batch calculator for technical indicators.

    Features:
    - Efficient batch calculation using 'ta' library
    - Automatic indicator selection based on available data
    - Optional normalization
    - Caching of calculated values
    """

    def __init__(self, universe: IndicatorUniverse = None,
                 normalizer: IndicatorNormalizer = None):
        """
        Initialize calculator.

        Args:
            universe: Indicator universe with definitions
            normalizer: Normalizer for converting to [-1, 1]
        """
        self.universe = universe or IndicatorUniverse()
        self.universe.load_all()

        self.normalizer = normalizer or IndicatorNormalizer(
            config=NormalizationConfig(),
            universe=self.universe
        )

        self._cache: Dict[str, pd.DataFrame] = {}

    def calculate_all(self, ohlcv: pd.DataFrame,
                      indicators: List[str] = None,
                      normalize: bool = False) -> pd.DataFrame:
        """
        Calculate all specified indicators.

        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume]
            indicators: List of indicator names (default: all)
            normalize: Whether to normalize results to [-1, 1]

        Returns:
            DataFrame with indicator columns
        """
        if ta is None:
            raise ImportError("'ta' library required. Install with: pip install ta")

        # Ensure column names are lowercase
        df = ohlcv.copy()
        df.columns = df.columns.str.lower()

        # Validate required columns
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        has_volume = 'volume' in df.columns

        # Get indicators to calculate
        if indicators is None:
            indicators = self.universe.get_all()

        # Filter out indicators that require volume if not available
        if not has_volume:
            valid_indicators = []
            for name in indicators:
                defn = self.universe.get_definition(name)
                if defn and not defn.requires_volume:
                    valid_indicators.append(name)
            indicators = valid_indicators

        # Calculate indicators
        results = pd.DataFrame(index=df.index)

        for name in indicators:
            try:
                defn = self.universe.get_definition(name)
                if not defn:
                    continue

                calc_result = self._calculate_single(df, defn)
                if calc_result is not None:
                    for col in calc_result.columns:
                        results[col] = calc_result[col]
            except Exception as e:
                logger.debug(f"Failed to calculate {name}: {e}")

        # Optionally normalize
        if normalize and not results.empty:
            results = self.normalizer.normalize_all(
                results,
                price_series=df['close']
            )

        return results

    def calculate_subset(self, ohlcv: pd.DataFrame,
                         indicator_names: List[str],
                         normalize: bool = False) -> pd.DataFrame:
        """
        Calculate a specific subset of indicators.
        """
        return self.calculate_all(ohlcv, indicator_names, normalize)

    def _calculate_single(self, df: pd.DataFrame,
                          defn: IndicatorDefinition) -> Optional[pd.DataFrame]:
        """Calculate a single indicator using 'ta' library."""
        if ta is None:
            return None

        base_name = defn.base_name
        p = defn.params
        res = pd.DataFrame(index=df.index)

        # Mapping 'pandas-ta' base_names to 'ta' library calls
        try:
            if base_name == 'rsi':
                indicator = momentum.RSIIndicator(close=df['close'], window=p.get('length', 14))
                res[defn.output_columns[0]] = indicator.rsi()

            elif base_name == 'macd':
                indicator = trend.MACD(
                    close=df['close'], 
                    window_slow=p.get('slow', 26), 
                    window_fast=p.get('fast', 12), 
                    window_sign=p.get('signal', 9)
                )
                # Output mapping depends on definition, usually main, signal, hist
                # In universe: MACD_f_s_sig, MACDh.., MACDs..
                cols = defn.output_columns
                if len(cols) >= 3:
                     # Standard MACD returns line, signal, diff (hist)
                     res[cols[0]] = indicator.macd()
                     res[cols[1]] = indicator.macd_diff() # Histogram often denoted as 'h'
                     res[cols[2]] = indicator.macd_signal()
                else:
                    res[cols[0]] = indicator.macd()

            elif base_name == 'stoch':
                indicator = momentum.StochasticOscillator(
                    high=df['high'], low=df['low'], close=df['close'],
                    window=p.get('k', 14), smooth_window=p.get('d', 3)
                )
                cols = defn.output_columns
                if len(cols) >= 2:
                    res[cols[0]] = indicator.stoch()
                    res[cols[1]] = indicator.stoch_signal()

            elif base_name == 'sma':
                indicator = trend.SMAIndicator(close=df['close'], window=p.get('length', 14))
                res[defn.output_columns[0]] = indicator.sma_indicator()

            elif base_name == 'ema':
                indicator = trend.EMAIndicator(close=df['close'], window=p.get('length', 14))
                res[defn.output_columns[0]] = indicator.ema_indicator()
                
            elif base_name == 'bbands':
                indicator = volatility.BollingerBands(
                    close=df['close'], window=p.get('length', 20), window_dev=p.get('std', 2)
                )
                # cols: L, M, U, B, P
                # ta gives: lband, mband, hband, wband, pband
                cols = defn.output_columns
                if len(cols) >= 3:
                    res[cols[0]] = indicator.bollinger_lband()
                    res[cols[1]] = indicator.bollinger_mavg()
                    res[cols[2]] = indicator.bollinger_hband()
                if len(cols) >= 5:
                    res[cols[3]] = indicator.bollinger_wband()
                    res[cols[4]] = indicator.bollinger_pband()

            elif base_name == 'atr':
                indicator = volatility.AverageTrueRange(
                    high=df['high'], low=df['low'], close=df['close'], window=p.get('length', 14)
                )
                res[defn.output_columns[0]] = indicator.average_true_range()

            elif base_name == 'adx':
                indicator = trend.ADXIndicator(
                    high=df['high'], low=df['low'], close=df['close'], window=p.get('length', 14)
                )
                cols = defn.output_columns
                if len(cols) >= 3:
                    res[cols[0]] = indicator.adx()
                    res[cols[1]] = indicator.adx_pos()
                    res[cols[2]] = indicator.adx_neg()

            elif base_name == 'obv':
                if 'volume' in df.columns:
                    indicator = volume.OnBalanceVolumeIndicator(
                        close=df['close'], volume=df['volume']
                    )
                    res[defn.output_columns[0]] = indicator.on_balance_volume()

            elif base_name == 'roc':
                indicator = momentum.ROCIndicator(close=df['close'], window=p.get('length', 12))
                res[defn.output_columns[0]] = indicator.roc()

            elif base_name == 'ichimoku':
                indicator = trend.IchimokuIndicator(
                    high=df['high'], low=df['low'], 
                    window1=p.get('n1', 9), window2=p.get('n2', 26), window3=p.get('n3', 52)
                )
                res[defn.output_columns[0]] = indicator.ichimoku_a()
                res[defn.output_columns[1]] = indicator.ichimoku_b()
                res[defn.output_columns[2]] = indicator.ichimoku_base_line()
                res[defn.output_columns[3]] = indicator.ichimoku_conversion_line()
                # ICS (Lagging Span) is usually close shifted back, ta doesn't always provide it directly in same way
                # We can compute it manually or skip. Let's skip for now or use close.shift(-26)
                # For safety, let's just use the 4 provided.
            
            elif base_name == 'vortex':
                indicator = trend.VortexIndicator(
                    high=df['high'], low=df['low'], close=df['close'], window=p.get('length', 14)
                )
                res[defn.output_columns[0]] = indicator.vortex_indicator_pos()
                res[defn.output_columns[1]] = indicator.vortex_indicator_neg()

            elif base_name == 'kst':
                indicator = trend.KSTIndicator(
                    close=df['close'], 
                    roc1=p.get('roc1', 10), roc2=p.get('roc2', 15), 
                    roc3=p.get('roc3', 20), roc4=p.get('roc4', 30),
                    window1=p.get('sma1', 10), window2=p.get('sma2', 10),
                    window3=p.get('sma3', 10), window4=p.get('sma4', 15)
                )
                res[defn.output_columns[0]] = indicator.kst()
                res[defn.output_columns[1]] = indicator.kst_sig()

            elif base_name == 'mass_index':
                # ta.trend.MassIndex
                indicator = trend.MassIndex(
                    high=df['high'], low=df['low'], 
                    window_fast=p.get('fast', 9), window_slow=p.get('slow', 25)
                )
                res[defn.output_columns[0]] = indicator.mass_index()

            elif base_name == 'coppock':
                # ta.momentum.ROCIndicator ? No, ta doesn't have Coppock explicit class usually?
                # Wait, ta library DOES NOT have Coppock Curve in standard classes commonly.
                # I should check if it exists. If not, I can implement manually: WMA(10) of (ROC(14) + ROC(11))
                # Let's check imports. ta.trend? ta.momentum?
                # Assuming ta library has it or I implement manual formula.
                # Formula: WMA(10, ROC(14) + ROC(11)).
                roc1 = momentum.ROCIndicator(close=df['close'], window=p.get('slow', 14)).roc()
                roc2 = momentum.ROCIndicator(close=df['close'], window=p.get('fast', 11)).roc()
                # WMA 10 on (roc1 + roc2)
                # Custom WMA function? or use ta.trend.WMAIndicator
                wma = trend.WMAIndicator(close=roc1 + roc2, window=p.get('length', 10))
                res[defn.output_columns[0]] = wma.wma_indicator()

            else:
                # Unsupported indicator in 'ta' mapping - just return None
                return None
                
            # Fill NaNs for safety
            res = res.bfill().ffill()
            return res

        except Exception as e:
            # logger.warning(f"Error calculating {defn.name} ({base_name}): {e}")
            return None

    def calculate_at_timestamp(self, ohlcv: pd.DataFrame,
                               indicator_names: List[str],
                               timestamp: pd.Timestamp,
                               normalize: bool = True) -> Dict[str, float]:
        """Calculate indicators at a specific timestamp."""
        if timestamp not in ohlcv.index:
            return {}
        historical = ohlcv.loc[:timestamp].copy()
        results = self.calculate_subset(historical, indicator_names, normalize=False)
        if results.empty:
            return {}
        values = {}
        for col in results.columns:
            value = results.loc[timestamp, col] if timestamp in results.index else np.nan
            if pd.notna(value):
                if normalize:
                    value = self.normalizer.normalize_at_timestamp(
                        results[col], col, timestamp
                    )
                values[col] = value
        return values


class QuickCalculator:
    """Simplified calculator using 'ta'."""
    @staticmethod
    def basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if ta is None: raise ImportError("'ta' required")
        df = df.copy()
        df.columns = df.columns.str.lower()
        results = pd.DataFrame(index=df.index)
        
        # RSI 14
        results['RSI_14'] = momentum.RSIIndicator(df['close'], 14).rsi()
        
        # MACD
        m = trend.MACD(df['close'])
        results['MACD_12_26_9'] = m.macd()
        results['MACDs_12_26_9'] = m.macd_signal()
        results['MACDh_12_26_9'] = m.macd_diff()
        
        return results

    @staticmethod
    def momentum_set(df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder
        return QuickCalculator.basic_indicators(df)

    @staticmethod
    def trend_set(df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder
        return QuickCalculator.basic_indicators(df)
