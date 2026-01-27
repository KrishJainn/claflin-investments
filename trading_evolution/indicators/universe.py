"""
Indicator Universe module - loads and manages 130+ indicators from pandas-ta.

This module provides:
- Lazy loading of indicator definitions
- Category-based organization
- Parameter configurations for each indicator
- Natural ranges for bounded indicators
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Callable, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class IndicatorDefinition:
    """Metadata for a single indicator."""
    name: str  # Unique identifier (e.g., 'RSI_14')
    base_name: str  # Base indicator name (e.g., 'RSI')
    category: str  # momentum, trend, volatility, volume, overlap, other
    params: Dict[str, Any] = field(default_factory=dict)
    output_columns: List[str] = field(default_factory=list)  # Output column names
    natural_range: Optional[Tuple[float, float]] = None  # (min, max) if bounded
    requires_volume: bool = False
    min_periods: int = 14  # Minimum data points needed


class IndicatorUniverse:
    """
    Manages the universe of 130+ technical indicators from pandas-ta.

    Organizes indicators by category:
    - Momentum: RSI, MACD, Stochastic, etc.
    - Trend: SMA, EMA, ADX, SuperTrend, etc.
    - Volatility: Bollinger Bands, ATR, Keltner, etc.
    - Volume: OBV, CMF, MFI, etc.
    - Overlap: Moving averages that overlay price
    """

    def __init__(self):
        """Initialize the indicator universe."""
        self._registry: Dict[str, IndicatorDefinition] = {}
        self._by_category: Dict[str, Set[str]] = {}
        self._loaded = False

    def load_all(self):
        """Load all available indicators from pandas-ta."""
        if self._loaded:
            return

        self._load_momentum_indicators()
        self._load_trend_indicators()
        self._load_volatility_indicators()
        self._load_volume_indicators()
        self._load_overlap_indicators()
        self._load_other_indicators()

        self._loaded = True
        logger.info(f"Loaded {len(self._registry)} indicators")

    def _register(self, indicator: IndicatorDefinition):
        """Register an indicator."""
        self._registry[indicator.name] = indicator

        if indicator.category not in self._by_category:
            self._by_category[indicator.category] = set()
        self._by_category[indicator.category].add(indicator.name)

    def _load_momentum_indicators(self):
        """Load momentum indicators."""
        # RSI variants
        for length in [7, 14, 21]:
            self._register(IndicatorDefinition(
                name=f'RSI_{length}',
                base_name='rsi',
                category='momentum',
                params={'length': length},
                output_columns=[f'RSI_{length}'],
                natural_range=(0, 100),
                min_periods=length + 1
            ))

        # MACD variants
        for fast, slow, signal in [(12, 26, 9), (8, 17, 9), (5, 35, 5)]:
            name = f'MACD_{fast}_{slow}_{signal}'
            self._register(IndicatorDefinition(
                name=name,
                base_name='macd',
                category='momentum',
                params={'fast': fast, 'slow': slow, 'signal': signal},
                output_columns=[f'MACD_{fast}_{slow}_{signal}',
                               f'MACDh_{fast}_{slow}_{signal}',
                               f'MACDs_{fast}_{slow}_{signal}'],
                min_periods=slow + signal
            ))

        # Stochastic
        for k, d in [(14, 3), (5, 3), (21, 5)]:
            self._register(IndicatorDefinition(
                name=f'STOCH_{k}_{d}',
                base_name='stoch',
                category='momentum',
                params={'k': k, 'd': d},
                output_columns=[f'STOCHk_{k}_{d}_3', f'STOCHd_{k}_{d}_3'],
                natural_range=(0, 100),
                min_periods=k + d
            ))

        # Williams %R
        for length in [14, 28]:
            self._register(IndicatorDefinition(
                name=f'WILLR_{length}',
                base_name='willr',
                category='momentum',
                params={'length': length},
                output_columns=[f'WILLR_{length}'],
                natural_range=(-100, 0),
                min_periods=length
            ))

        # CCI
        for length in [14, 20]:
            self._register(IndicatorDefinition(
                name=f'CCI_{length}',
                base_name='cci',
                category='momentum',
                params={'length': length},
                output_columns=[f'CCI_{length}_0.015'],
                min_periods=length
            ))

        # ROC (Rate of Change)
        for length in [10, 20]:
            self._register(IndicatorDefinition(
                name=f'ROC_{length}',
                base_name='roc',
                category='momentum',
                params={'length': length},
                output_columns=[f'ROC_{length}'],
                min_periods=length + 1
            ))

        # MOM (Momentum)
        for length in [10, 20]:
            self._register(IndicatorDefinition(
                name=f'MOM_{length}',
                base_name='mom',
                category='momentum',
                params={'length': length},
                output_columns=[f'MOM_{length}'],
                min_periods=length + 1
            ))

        # CMO (Chande Momentum Oscillator)
        for length in [14, 20]:
            self._register(IndicatorDefinition(
                name=f'CMO_{length}',
                base_name='cmo',
                category='momentum',
                params={'length': length},
                output_columns=[f'CMO_{length}'],
                natural_range=(-100, 100),
                min_periods=length + 1
            ))

        # TSI (True Strength Index)
        self._register(IndicatorDefinition(
            name='TSI_13_25',
            base_name='tsi',
            category='momentum',
            params={'fast': 13, 'slow': 25},
            output_columns=['TSI_13_25_13', 'TSIs_13_25_13'],
            natural_range=(-100, 100),
            min_periods=40
        ))

        # UO (Ultimate Oscillator)
        self._register(IndicatorDefinition(
            name='UO_7_14_28',
            base_name='uo',
            category='momentum',
            params={'fast': 7, 'medium': 14, 'slow': 28},
            output_columns=['UO_7_14_28'],
            natural_range=(0, 100),
            min_periods=28
        ))

        # AO (Awesome Oscillator)
        self._register(IndicatorDefinition(
            name='AO_5_34',
            base_name='ao',
            category='momentum',
            params={'fast': 5, 'slow': 34},
            output_columns=['AO_5_34'],
            min_periods=34
        ))

    def _load_trend_indicators(self):
        """Load trend indicators."""
        # ADX
        for length in [14, 20]:
            self._register(IndicatorDefinition(
                name=f'ADX_{length}',
                base_name='adx',
                category='trend',
                params={'length': length},
                output_columns=[f'ADX_{length}', f'DMP_{length}', f'DMN_{length}'],
                natural_range=(0, 100),
                min_periods=length * 2
            ))

        # Aroon
        for length in [14, 25]:
            self._register(IndicatorDefinition(
                name=f'AROON_{length}',
                base_name='aroon',
                category='trend',
                params={'length': length},
                output_columns=[f'AROOND_{length}', f'AROONU_{length}', f'AROONOSC_{length}'],
                natural_range=(0, 100),
                min_periods=length + 1
            ))

        # SuperTrend
        for length, mult in [(7, 3), (10, 2), (20, 3)]:
            self._register(IndicatorDefinition(
                name=f'SUPERTREND_{length}_{mult}',
                base_name='supertrend',
                category='trend',
                params={'length': length, 'multiplier': mult},
                output_columns=[f'SUPERT_{length}_{float(mult)}', f'SUPERTd_{length}_{float(mult)}',
                               f'SUPERTl_{length}_{float(mult)}', f'SUPERTs_{length}_{float(mult)}'],
                min_periods=length
            ))

        # PSAR (Parabolic SAR)
        self._register(IndicatorDefinition(
            name='PSAR',
            base_name='psar',
            category='trend',
            params={'af0': 0.02, 'af': 0.02, 'max_af': 0.2},
            output_columns=['PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2',
                           'PSARr_0.02_0.2'],
            min_periods=2
        ))

        # Linear Regression Slope
        for length in [14, 25]:
            self._register(IndicatorDefinition(
                name=f'LINREG_SLOPE_{length}',
                base_name='linreg',
                category='trend',
                params={'length': length},
                output_columns=[f'LR_{length}', f'LRr_{length}', f'LRm_{length}', f'LRb_{length}'],
                min_periods=length
            ))

    def _load_volatility_indicators(self):
        """Load volatility indicators."""
        # ATR
        for length in [14, 20]:
            self._register(IndicatorDefinition(
                name=f'ATR_{length}',
                base_name='atr',
                category='volatility',
                params={'length': length},
                output_columns=[f'ATRr_{length}'],
                min_periods=length
            ))

        # NATR (Normalized ATR)
        for length in [14, 20]:
            self._register(IndicatorDefinition(
                name=f'NATR_{length}',
                base_name='natr',
                category='volatility',
                params={'length': length},
                output_columns=[f'NATR_{length}'],
                min_periods=length
            ))

        # Bollinger Bands
        for length, std in [(20, 2), (20, 2.5), (10, 1.5)]:
            self._register(IndicatorDefinition(
                name=f'BBANDS_{length}_{std}',
                base_name='bbands',
                category='volatility',
                params={'length': length, 'std': std},
                output_columns=[f'BBL_{length}_{float(std)}', f'BBM_{length}_{float(std)}',
                               f'BBU_{length}_{float(std)}', f'BBB_{length}_{float(std)}',
                               f'BBP_{length}_{float(std)}'],
                min_periods=length
            ))

        # Keltner Channels
        for length, mult in [(20, 2), (20, 1.5)]:
            self._register(IndicatorDefinition(
                name=f'KC_{length}_{mult}',
                base_name='kc',
                category='volatility',
                params={'length': length, 'scalar': mult},
                output_columns=[f'KCLe_{length}_{float(mult)}', f'KCBe_{length}_{float(mult)}',
                               f'KCUe_{length}_{float(mult)}'],
                min_periods=length
            ))

        # Donchian Channels
        for length in [20, 50]:
            self._register(IndicatorDefinition(
                name=f'DONCHIAN_{length}',
                base_name='donchian',
                category='volatility',
                params={'lower_length': length, 'upper_length': length},
                output_columns=[f'DCL_{length}_{length}', f'DCM_{length}_{length}',
                               f'DCU_{length}_{length}'],
                min_periods=length
            ))

        # True Range
        self._register(IndicatorDefinition(
            name='TRUERANGE',
            base_name='true_range',
            category='volatility',
            params={},
            output_columns=['TRUERANGE'],
            min_periods=1
        ))

    def _load_volume_indicators(self):
        """Load volume indicators."""
        # OBV (On Balance Volume)
        self._register(IndicatorDefinition(
            name='OBV',
            base_name='obv',
            category='volume',
            params={},
            output_columns=['OBV'],
            requires_volume=True,
            min_periods=1
        ))

        # AD (Accumulation/Distribution)
        self._register(IndicatorDefinition(
            name='AD',
            base_name='ad',
            category='volume',
            params={},
            output_columns=['AD'],
            requires_volume=True,
            min_periods=1
        ))

        # ADOSC (Chaikin A/D Oscillator)
        self._register(IndicatorDefinition(
            name='ADOSC_3_10',
            base_name='adosc',
            category='volume',
            params={'fast': 3, 'slow': 10},
            output_columns=['ADOSC_3_10'],
            requires_volume=True,
            min_periods=10
        ))

        # CMF (Chaikin Money Flow)
        for length in [20, 21]:
            self._register(IndicatorDefinition(
                name=f'CMF_{length}',
                base_name='cmf',
                category='volume',
                params={'length': length},
                output_columns=[f'CMF_{length}'],
                natural_range=(-1, 1),
                requires_volume=True,
                min_periods=length
            ))

        # MFI (Money Flow Index)
        for length in [14, 20]:
            self._register(IndicatorDefinition(
                name=f'MFI_{length}',
                base_name='mfi',
                category='volume',
                params={'length': length},
                output_columns=[f'MFI_{length}'],
                natural_range=(0, 100),
                requires_volume=True,
                min_periods=length + 1
            ))

        # EFI (Elder's Force Index)
        for length in [13, 20]:
            self._register(IndicatorDefinition(
                name=f'EFI_{length}',
                base_name='efi',
                category='volume',
                params={'length': length},
                output_columns=[f'EFI_{length}'],
                requires_volume=True,
                min_periods=length
            ))

        # VWAP (Volume Weighted Average Price) - note: intraday only
        # self._register(IndicatorDefinition(
        #     name='VWAP',
        #     base_name='vwap',
        #     category='volume',
        #     params={},
        #     output_columns=['VWAP_D'],
        #     requires_volume=True,
        #     min_periods=1
        # ))

        # PVI/NVI (Positive/Negative Volume Index)
        self._register(IndicatorDefinition(
            name='PVI',
            base_name='pvi',
            category='volume',
            params={},
            output_columns=['PVI'],
            requires_volume=True,
            min_periods=1
        ))

        self._register(IndicatorDefinition(
            name='NVI',
            base_name='nvi',
            category='volume',
            params={},
            output_columns=['NVI'],
            requires_volume=True,
            min_periods=1
        ))

    def _load_overlap_indicators(self):
        """Load overlap (moving average) indicators."""
        # SMA
        for length in [10, 20, 50, 100, 200]:
            self._register(IndicatorDefinition(
                name=f'SMA_{length}',
                base_name='sma',
                category='overlap',
                params={'length': length},
                output_columns=[f'SMA_{length}'],
                min_periods=length
            ))

        # EMA
        for length in [10, 20, 50, 100, 200]:
            self._register(IndicatorDefinition(
                name=f'EMA_{length}',
                base_name='ema',
                category='overlap',
                params={'length': length},
                output_columns=[f'EMA_{length}'],
                min_periods=length
            ))

        # WMA (Weighted Moving Average)
        for length in [10, 20]:
            self._register(IndicatorDefinition(
                name=f'WMA_{length}',
                base_name='wma',
                category='overlap',
                params={'length': length},
                output_columns=[f'WMA_{length}'],
                min_periods=length
            ))

        # DEMA (Double EMA)
        for length in [10, 20]:
            self._register(IndicatorDefinition(
                name=f'DEMA_{length}',
                base_name='dema',
                category='overlap',
                params={'length': length},
                output_columns=[f'DEMA_{length}'],
                min_periods=length * 2
            ))

        # TEMA (Triple EMA)
        for length in [10, 20]:
            self._register(IndicatorDefinition(
                name=f'TEMA_{length}',
                base_name='tema',
                category='overlap',
                params={'length': length},
                output_columns=[f'TEMA_{length}'],
                min_periods=length * 3
            ))

        # KAMA (Kaufman Adaptive MA)
        for length in [10, 20]:
            self._register(IndicatorDefinition(
                name=f'KAMA_{length}',
                base_name='kama',
                category='overlap',
                params={'length': length},
                output_columns=[f'KAMA_{length}_2_30'],
                min_periods=length
            ))

        # T3 (Tillson T3)
        for length in [5, 10]:
            self._register(IndicatorDefinition(
                name=f'T3_{length}',
                base_name='t3',
                category='overlap',
                params={'length': length},
                output_columns=[f'T3_{length}_0.7'],
                min_periods=length * 6
            ))

        # HMA (Hull MA)
        for length in [9, 16]:
            self._register(IndicatorDefinition(
                name=f'HMA_{length}',
                base_name='hma',
                category='overlap',
                params={'length': length},
                output_columns=[f'HMA_{length}'],
                min_periods=length
            ))

        # VWMA (Volume Weighted MA)
        for length in [10, 20]:
            self._register(IndicatorDefinition(
                name=f'VWMA_{length}',
                base_name='vwma',
                category='overlap',
                params={'length': length},
                output_columns=[f'VWMA_{length}'],
                requires_volume=True,
                min_periods=length
            ))

    def _load_other_indicators(self):
        """Load other/miscellaneous indicators."""
        # Percent B (related to Bollinger)
        # Already included as BBP in bollinger bands

        # Pivot Points
        self._register(IndicatorDefinition(
            name='PIVOTS',
            base_name='pivots',
            category='other',
            params={},
            output_columns=['S1', 'S2', 'S3', 'P', 'R1', 'R2', 'R3'],
            min_periods=1
        ))

        # ZScore
        for length in [20, 50]:
            self._register(IndicatorDefinition(
                name=f'ZSCORE_{length}',
                base_name='zscore',
                category='other',
                params={'length': length},
                output_columns=[f'ZS_{length}'],
                min_periods=length
            ))

        # Correlation with SPY (placeholder - would need SPY data)
        # This is more complex and would need modification

        # Ichimoku Cloud (Trend)
        self._register(IndicatorDefinition(
            name='ICHIMOKU',
            base_name='ichimoku',
            category='trend',
            params={'n1': 9, 'n2': 26, 'n3': 52},
            output_columns=['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26'],
            min_periods=52
        ))

        # Vortex Indicator (Trend)
        self._register(IndicatorDefinition(
            name='VORTEX_14',
            base_name='vortex',
            category='trend',
            params={'length': 14},
            output_columns=['VTXP_14', 'VTXM_14'],
            natural_range=(0, 2), # Typically around 1
            min_periods=14
        ))

        # KST Oscillator (Momentum)
        self._register(IndicatorDefinition(
            name='KST',
            base_name='kst',
            category='momentum',
            params={'roc1': 10, 'roc2': 15, 'roc3': 20, 'roc4': 30,
                    'sma1': 10, 'sma2': 10, 'sma3': 10, 'sma4': 15},
            output_columns=['KST_10_15_20_30_10_10_10_15', 'KSTs_9'],
            natural_range=(-100, 100),
            min_periods=45
        ))

        # Mass Index (Volatility/Reversal)
        self._register(IndicatorDefinition(
            name='MASS_INDEX',
            base_name='mass_index',
            category='volatility', # Often grouped here or other
            params={'fast': 9, 'slow': 25},
            output_columns=['MASSI_9_25'],
            min_periods=25
        ))

        # Coppock Curve (Momentum)
        self._register(IndicatorDefinition(
            name='COPPOCK',
            base_name='coppock',
            category='momentum',
            params={'fast': 11, 'slow': 14, 'length': 10},
            output_columns=['COPC_11_14_10'],
            min_periods=24
        ))

    # Public API methods
    def get_all(self) -> List[str]:
        """Get all registered indicator names."""
        if not self._loaded:
            self.load_all()
        return list(self._registry.keys())

    def get_by_category(self, category: str) -> List[str]:
        """Get indicators by category."""
        if not self._loaded:
            self.load_all()
        return list(self._by_category.get(category, set()))

    def get_definition(self, name: str) -> Optional[IndicatorDefinition]:
        """Get indicator definition by name."""
        if not self._loaded:
            self.load_all()
        return self._registry.get(name)

    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        if not self._loaded:
            self.load_all()
        return list(self._by_category.keys())

    @property
    def total_count(self) -> int:
        """Get total number of registered indicators."""
        if not self._loaded:
            self.load_all()
        return len(self._registry)

    def get_summary(self) -> Dict[str, int]:
        """Get count of indicators per category."""
        if not self._loaded:
            self.load_all()
        return {cat: len(inds) for cat, inds in self._by_category.items()}
