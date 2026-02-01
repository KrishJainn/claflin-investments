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

        # Build output-column → DNA-name mapping for primary columns
        self._col_to_dna = self._build_column_mapping()

    def _build_column_mapping(self) -> Dict[str, str]:
        """
        Build mapping: output_column_name → indicator_name (DNA name).

        For multi-output indicators we pick one "primary" column that is
        most useful as a standalone signal:
          - BBANDS → BBP (percent-b is the signal column)
          - STOCH → STOCHk (the fast line)
          - ADX → ADX (the main ADX line)
          - AROON → AROONOSC (oscillator)
          - SUPERTREND → SUPERTd (direction)
          - MACD → MACD line
          - KST → KST line
          - ICHIMOKU → ISA (span A)
        For single-output indicators the mapping is direct.
        """
        mapping: Dict[str, str] = {}
        for name in self.universe.get_all():
            defn = self.universe.get_definition(name)
            if not defn:
                continue

            cols = defn.output_columns
            if not cols:
                continue

            # For multi-column indicators, pick the primary signal column
            if defn.base_name == 'bbands':
                # BBP (percent-b) is the signal column
                for c in cols:
                    if c.startswith('BBP'):
                        mapping[c] = name
                        break
            elif defn.base_name == 'stoch':
                # STOCHk is the primary line
                for c in cols:
                    if c.startswith('STOCHk'):
                        mapping[c] = name
                        break
            elif defn.base_name == 'adx':
                # ADX itself
                for c in cols:
                    if c.startswith('ADX'):
                        mapping[c] = name
                        break
            elif defn.base_name == 'aroon':
                # AROONOSC
                for c in cols:
                    if c.startswith('AROONOSC'):
                        mapping[c] = name
                        break
            elif defn.base_name == 'supertrend':
                # SUPERTd (direction is the signal)
                for c in cols:
                    if c.startswith('SUPERTd'):
                        mapping[c] = name
                        break
            elif defn.base_name == 'macd':
                # MACD line
                mapping[cols[0]] = name
            elif defn.base_name == 'kst':
                # KST line
                mapping[cols[0]] = name
            elif defn.base_name == 'ichimoku':
                # ISA (span A) as primary
                mapping[cols[0]] = name
            elif defn.base_name == 'vortex':
                # VTXP as primary
                mapping[cols[0]] = name
            elif defn.base_name == 'psar':
                mapping[cols[0]] = name
            elif defn.base_name == 'linreg':
                # LR (the regression line) as primary
                mapping[cols[0]] = name
            elif defn.base_name == 'kc':
                # KCBe (middle band)
                for c in cols:
                    if 'Be' in c or 'B' in c:
                        mapping[c] = name
                        break
                else:
                    mapping[cols[0]] = name
            elif defn.base_name == 'donchian':
                # DCM (middle)
                for c in cols:
                    if 'DCM' in c:
                        mapping[c] = name
                        break
                else:
                    mapping[cols[0]] = name
            elif defn.base_name == 'pivots':
                # P (pivot point itself)
                for c in cols:
                    if c == 'P':
                        mapping[c] = name
                        break
            else:
                # Single-output: first (and usually only) column
                mapping[cols[0]] = name

        return mapping

    def rename_to_dna_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename output columns to match DNA indicator names.

        This resolves the mismatch between calculator output column names
        (e.g. ATRr_14, BBP_20_2.5, STOCHk_5_3_3) and DNA indicator names
        (e.g. ATR_14, BBANDS_20_2.5, STOCH_5_3).

        Only primary columns are renamed; secondary columns are dropped.
        """
        rename_map = {}
        keep_cols = set()
        for col in df.columns:
            if col in self._col_to_dna:
                dna_name = self._col_to_dna[col]
                rename_map[col] = dna_name
                keep_cols.add(col)
            # Also keep columns that already match a DNA name exactly
            elif col in [d.name for d in self.universe._registry.values()]:
                keep_cols.add(col)

        # Keep only primary columns + already-matching columns
        result = df[list(keep_cols)].copy()
        result = result.rename(columns=rename_map)
        return result

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
                roc1 = momentum.ROCIndicator(close=df['close'], window=p.get('slow', 14)).roc()
                roc2 = momentum.ROCIndicator(close=df['close'], window=p.get('fast', 11)).roc()
                wma_ind = trend.WMAIndicator(close=roc1 + roc2, window=p.get('length', 10))
                res[defn.output_columns[0]] = wma_ind.wma()

            elif base_name == 'tsi':
                indicator = momentum.TSIIndicator(
                    close=df['close'],
                    window_slow=p.get('slow', 25),
                    window_fast=p.get('fast', 13),
                )
                cols = defn.output_columns
                res[cols[0]] = indicator.tsi()

            elif base_name == 'ao':
                indicator = momentum.AwesomeOscillatorIndicator(
                    high=df['high'], low=df['low'],
                    window1=p.get('fast', 5), window2=p.get('slow', 34),
                )
                res[defn.output_columns[0]] = indicator.awesome_oscillator()

            elif base_name == 'cci':
                indicator = trend.CCIIndicator(
                    high=df['high'], low=df['low'], close=df['close'],
                    window=p.get('length', 20),
                )
                res[defn.output_columns[0]] = indicator.cci()

            elif base_name == 'willr':
                indicator = momentum.WilliamsRIndicator(
                    high=df['high'], low=df['low'], close=df['close'],
                    lbp=p.get('length', 14),
                )
                res[defn.output_columns[0]] = indicator.williams_r()

            elif base_name == 'mom':
                # Momentum = close - close.shift(n)
                length = p.get('length', 10)
                res[defn.output_columns[0]] = df['close'] - df['close'].shift(length)

            elif base_name == 'cmo':
                # Chande Momentum Oscillator (manual calculation)
                length = p.get('length', 14)
                diff = df['close'].diff()
                up = diff.clip(lower=0).rolling(length).sum()
                down = (-diff.clip(upper=0)).rolling(length).sum()
                res[defn.output_columns[0]] = ((up - down) / (up + down)) * 100

            elif base_name == 'aroon':
                indicator = trend.AroonIndicator(
                    high=df['high'], low=df['low'],
                    window=p.get('length', 14),
                )
                cols = defn.output_columns
                if len(cols) >= 3:
                    res[cols[0]] = indicator.aroon_down()
                    res[cols[1]] = indicator.aroon_up()
                    res[cols[2]] = indicator.aroon_up() - indicator.aroon_down()

            elif base_name == 'supertrend':
                # SuperTrend using ATR bands
                length = p.get('length', 7)
                mult = p.get('multiplier', 3)
                atr_ind = volatility.AverageTrueRange(
                    high=df['high'], low=df['low'], close=df['close'], window=length,
                )
                atr_vals = atr_ind.average_true_range()
                hl2 = (df['high'] + df['low']) / 2
                upper_band = hl2 + mult * atr_vals
                lower_band = hl2 - mult * atr_vals

                supertrend = pd.Series(0.0, index=df.index)
                direction = pd.Series(1, index=df.index)
                for i in range(1, len(df)):
                    if pd.isna(atr_vals.iloc[i]):
                        continue
                    if df['close'].iloc[i] > upper_band.iloc[i - 1]:
                        direction.iloc[i] = 1
                    elif df['close'].iloc[i] < lower_band.iloc[i - 1]:
                        direction.iloc[i] = -1
                    else:
                        direction.iloc[i] = direction.iloc[i - 1]

                    if direction.iloc[i] == 1:
                        supertrend.iloc[i] = lower_band.iloc[i]
                    else:
                        supertrend.iloc[i] = upper_band.iloc[i]

                cols = defn.output_columns
                res[cols[0]] = supertrend
                if len(cols) >= 2:
                    res[cols[1]] = direction.astype(float)

            elif base_name == 'cmf':
                if 'volume' in df.columns:
                    indicator = volume.ChaikinMoneyFlowIndicator(
                        high=df['high'], low=df['low'], close=df['close'],
                        volume=df['volume'], window=p.get('length', 20),
                    )
                    res[defn.output_columns[0]] = indicator.chaikin_money_flow()

            elif base_name == 'mfi':
                if 'volume' in df.columns:
                    indicator = volume.MFIIndicator(
                        high=df['high'], low=df['low'], close=df['close'],
                        volume=df['volume'], window=p.get('length', 14),
                    )
                    res[defn.output_columns[0]] = indicator.money_flow_index()

            elif base_name == 'efi':
                if 'volume' in df.columns:
                    indicator = volume.ForceIndexIndicator(
                        close=df['close'], volume=df['volume'],
                        window=p.get('length', 13),
                    )
                    res[defn.output_columns[0]] = indicator.force_index()

            elif base_name == 'nvi':
                if 'volume' in df.columns:
                    indicator = volume.NegativeVolumeIndexIndicator(
                        close=df['close'], volume=df['volume'],
                    )
                    res[defn.output_columns[0]] = indicator.negative_volume_index()

            elif base_name == 'pvi':
                if 'volume' in df.columns:
                    # Manual PVI: start at 1000, increase by pct change when volume > prev volume
                    pvi = pd.Series(1000.0, index=df.index)
                    for i in range(1, len(df)):
                        if df['volume'].iloc[i] > df['volume'].iloc[i - 1]:
                            pvi.iloc[i] = pvi.iloc[i - 1] * (1 + (df['close'].iloc[i] / df['close'].iloc[i - 1] - 1))
                        else:
                            pvi.iloc[i] = pvi.iloc[i - 1]
                    res[defn.output_columns[0]] = pvi

            elif base_name == 'wma':
                indicator = trend.WMAIndicator(
                    close=df['close'], window=p.get('length', 10),
                )
                res[defn.output_columns[0]] = indicator.wma()

            elif base_name == 'dema':
                # DEMA = 2*EMA(n) - EMA(EMA(n))
                length = p.get('length', 20)
                ema1 = trend.EMAIndicator(close=df['close'], window=length).ema_indicator()
                ema2 = trend.EMAIndicator(close=ema1, window=length).ema_indicator()
                res[defn.output_columns[0]] = 2 * ema1 - ema2

            elif base_name == 'tema':
                # TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
                length = p.get('length', 20)
                ema1 = trend.EMAIndicator(close=df['close'], window=length).ema_indicator()
                ema2 = trend.EMAIndicator(close=ema1, window=length).ema_indicator()
                ema3 = trend.EMAIndicator(close=ema2, window=length).ema_indicator()
                res[defn.output_columns[0]] = 3 * ema1 - 3 * ema2 + ema3

            elif base_name == 'vwma':
                if 'volume' in df.columns:
                    length = p.get('length', 10)
                    vp = df['close'] * df['volume']
                    res[defn.output_columns[0]] = vp.rolling(length).sum() / df['volume'].rolling(length).sum()

            elif base_name == 'hma':
                # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
                length = p.get('length', 9)
                half = max(int(length / 2), 1)
                sqrt_len = max(int(length ** 0.5), 1)
                wma1 = trend.WMAIndicator(close=df['close'], window=half).wma()
                wma2 = trend.WMAIndicator(close=df['close'], window=length).wma()
                diff_wma = 2 * wma1 - wma2
                hma = trend.WMAIndicator(close=diff_wma, window=sqrt_len).wma()
                res[defn.output_columns[0]] = hma

            elif base_name == 'kama':
                indicator = momentum.KAMAIndicator(
                    close=df['close'], window=p.get('length', 10),
                    pow1=2, pow2=30,
                )
                res[defn.output_columns[0]] = indicator.kama()

            elif base_name == 'natr':
                # NATR = ATR / close * 100
                length = p.get('length', 14)
                atr_ind = volatility.AverageTrueRange(
                    high=df['high'], low=df['low'], close=df['close'], window=length,
                )
                res[defn.output_columns[0]] = (atr_ind.average_true_range() / df['close']) * 100

            elif base_name == 'kc':
                indicator = volatility.KeltnerChannel(
                    high=df['high'], low=df['low'], close=df['close'],
                    window=p.get('length', 20), window_atr=p.get('length', 20),
                    multiplier=p.get('scalar', 2),
                )
                cols = defn.output_columns
                if len(cols) >= 3:
                    res[cols[0]] = indicator.keltner_channel_lband()
                    res[cols[1]] = indicator.keltner_channel_mband()
                    res[cols[2]] = indicator.keltner_channel_hband()

            elif base_name == 'donchian':
                indicator = volatility.DonchianChannel(
                    high=df['high'], low=df['low'], close=df['close'],
                    window=p.get('lower_length', 20),
                )
                cols = defn.output_columns
                if len(cols) >= 3:
                    res[cols[0]] = indicator.donchian_channel_lband()
                    res[cols[1]] = indicator.donchian_channel_mband()
                    res[cols[2]] = indicator.donchian_channel_hband()

            elif base_name == 'true_range':
                import numpy as _np
                prev_close = df['close'].shift(1)
                tr1 = df['high'] - df['low']
                tr2 = (df['high'] - prev_close).abs()
                tr3 = (df['low'] - prev_close).abs()
                res[defn.output_columns[0]] = _np.maximum(tr1, _np.maximum(tr2, tr3))

            elif base_name == 'psar':
                indicator = trend.PSARIndicator(
                    high=df['high'], low=df['low'], close=df['close'],
                    step=p.get('af', 0.02), max_step=p.get('max_af', 0.2),
                )
                cols = defn.output_columns
                res[cols[0]] = indicator.psar_up()
                if len(cols) >= 2:
                    res[cols[1]] = indicator.psar_down()
                if len(cols) >= 3:
                    res[cols[2]] = indicator.psar_up_indicator()
                if len(cols) >= 4:
                    res[cols[3]] = indicator.psar_down_indicator()

            elif base_name == 'ad':
                if 'volume' in df.columns:
                    indicator = volume.AccDistIndexIndicator(
                        high=df['high'], low=df['low'], close=df['close'],
                        volume=df['volume'],
                    )
                    res[defn.output_columns[0]] = indicator.acc_dist_index()

            elif base_name == 'adosc':
                if 'volume' in df.columns:
                    # A/D Oscillator = EMA(fast, A/D) - EMA(slow, A/D)
                    ad_ind = volume.AccDistIndexIndicator(
                        high=df['high'], low=df['low'], close=df['close'],
                        volume=df['volume'],
                    )
                    ad_line = ad_ind.acc_dist_index()
                    fast = p.get('fast', 3)
                    slow_p = p.get('slow', 10)
                    ema_fast = trend.EMAIndicator(close=ad_line, window=fast).ema_indicator()
                    ema_slow = trend.EMAIndicator(close=ad_line, window=slow_p).ema_indicator()
                    res[defn.output_columns[0]] = ema_fast - ema_slow

            elif base_name == 'uo':
                indicator = momentum.UltimateOscillator(
                    high=df['high'], low=df['low'], close=df['close'],
                    window1=p.get('fast', 7), window2=p.get('medium', 14),
                    window3=p.get('slow', 28),
                )
                res[defn.output_columns[0]] = indicator.ultimate_oscillator()

            elif base_name == 'linreg':
                # Linear regression slope
                length = p.get('length', 14)
                from numpy.polynomial import polynomial as P
                lr_vals = pd.Series(np.nan, index=df.index)
                for i in range(length, len(df)):
                    y = df['close'].iloc[i - length:i].values
                    x = np.arange(length)
                    try:
                        coeffs = np.polyfit(x, y, 1)
                        lr_vals.iloc[i] = coeffs[0]  # slope
                    except Exception:
                        pass
                cols = defn.output_columns
                res[cols[0]] = lr_vals

            elif base_name == 'zscore':
                length = p.get('length', 20)
                rolling_mean = df['close'].rolling(length).mean()
                rolling_std = df['close'].rolling(length).std()
                res[defn.output_columns[0]] = (df['close'] - rolling_mean) / rolling_std

            elif base_name == 'pivots':
                # Standard pivot points from previous bar
                pp = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
                cols = defn.output_columns
                # S1, S2, S3, P, R1, R2, R3
                if len(cols) >= 7:
                    res[cols[0]] = 2 * pp - df['high'].shift(1)   # S1
                    res[cols[1]] = pp - (df['high'].shift(1) - df['low'].shift(1))  # S2
                    res[cols[2]] = df['low'].shift(1) - 2 * (df['high'].shift(1) - pp)  # S3
                    res[cols[3]] = pp  # P
                    res[cols[4]] = 2 * pp - df['low'].shift(1)  # R1
                    res[cols[5]] = pp + (df['high'].shift(1) - df['low'].shift(1))  # R2
                    res[cols[6]] = df['high'].shift(1) + 2 * (pp - df['low'].shift(1))  # R3

            elif base_name == 't3':
                # Tillson T3 = EMA of EMA of ... (6 layers with volume factor)
                length = p.get('length', 5)
                vf = 0.7
                c1 = -(vf ** 3)
                c2 = 3 * vf ** 2 + 3 * vf ** 3
                c3 = -6 * vf ** 2 - 3 * vf - 3 * vf ** 3
                c4 = 1 + 3 * vf + vf ** 3 + 3 * vf ** 2
                e1 = trend.EMAIndicator(close=df['close'], window=length).ema_indicator()
                e2 = trend.EMAIndicator(close=e1, window=length).ema_indicator()
                e3 = trend.EMAIndicator(close=e2, window=length).ema_indicator()
                e4 = trend.EMAIndicator(close=e3, window=length).ema_indicator()
                e5 = trend.EMAIndicator(close=e4, window=length).ema_indicator()
                e6 = trend.EMAIndicator(close=e5, window=length).ema_indicator()
                res[defn.output_columns[0]] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

            else:
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
