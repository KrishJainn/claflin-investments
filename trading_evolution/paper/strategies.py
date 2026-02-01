"""
Best Performing Strategies.

Consolidated collection of the best-performing trading strategies
from backtesting and optimization runs.

SUMMARY:
- SAFE: 82% win rate, conservative
- BALANCED: 61% win rate, $136k profit, good risk/reward  
- AGGRESSIVE: 51% win rate, $221k profit, highest profit
- OPTIMIZED: Best parameters from grid search
"""

from typing import Dict, Any
from dataclasses import dataclass


# ============================================================================
# STRATEGY 1: SAFE (High Win Rate Conservative)
# ============================================================================
# Performance: 82% Win Rate, Lower profit but consistent
# Best for: Capital preservation, beginners, conservative accounts
SAFE_WEIGHTS = {
    "KST": 0.8221, "STOCH_21_5": 0.2018, "MACD_12_26_9": -0.0129, 
    "NVI": -0.0253, "CCI_14": -0.0655, "CMF_20": 0.7471, 
    "EFI_20": -0.1005, "SUPERTREND_7_3": -0.5066, "ADX_14": -0.4027, 
    "COPPOCK": -0.2199, "ZSCORE_50": 0.5693, "ZSCORE_20": 0.4706, 
    "HMA_9": 0.6080, "SMA_100": 0.3522, "HMA_16": 0.0587, 
    "RSI_7": -0.2635, "WMA_10": 0.0201, "NATR_14": 0.2203, 
    "TEMA_10": -0.0935, "VWMA_10": 0.2957, "UO_7_14_28": 0.2135, 
    "AD": -0.2890, "VORTEX_14": 0.3862, "MOM_10": 0.2177, 
    "MASS_INDEX": -0.2596, "NATR_20": -0.0185, "MOM_20": -0.0744, 
    "AROON_25": -0.1304, "BBANDS_10_1.5": -0.1702, "WMA_20": 0.2055, 
    "EMA_100": 0.0669, "DONCHIAN_20": -0.3375, "DEMA_20": 0.1058, 
    "ADX_20": -1.0, "T3_5": -0.3051, "DONCHIAN_50": -0.1684, 
    "KC_20_2": -0.0262, "T3_10": -0.2661, "STOCH_5_3": -0.2506, 
    "CMF_21": -0.2381, "TRUERANGE": -0.4556, "STOCH_14_3": -0.0527, 
    "BBANDS_20_2": 0.3865, "TEMA_20": 0.9034, "ICHIMOKU": -0.0120, 
    "EMA_20": 0.1507, "SUPERTREND_10_2": 0.2666, "SMA_200": -0.0373, 
    "KAMA_20": -0.0124, "CCI_20": 0.1167, "LINREG_SLOPE_14": -0.3821, 
    "CMO_20": 0.0353, "KC_20_1.5": -0.7425, "ATR_20": 0.1949, 
    "PIVOTS": -0.2605, "LINREG_SLOPE_25": 0.2104, "OBV": -0.2366, 
    "WILLR_28": 0.1063, "PVI": -0.4915
}


# ============================================================================
# STRATEGY 2: BALANCED (Good Risk/Reward)
# ============================================================================
# Performance: 61% Win Rate, $136k profit, good balance
# Best for: Most traders, balanced approach
BALANCED_WEIGHTS = {
    "SMA_10": -0.0132, "KC_20_2": -0.5488, "DEMA_10": -0.6213, 
    "DONCHIAN_20": 0.6942, "EFI_13": 0.2689, "AROON_25": -0.9460, 
    "KST": 0.9153, "CMF_20": 0.5052, "CMO_14": 0.1215, 
    "TSI_13_25": 0.0832, "WILLR_28": 0.8833, "ZSCORE_50": -0.1787, 
    "RSI_21": 0.1389, "STOCH_14_3": 0.0243, "SUPERTREND_10_2": -0.8235, 
    "EMA_50": 0.0959, "VWMA_10": 0.5291, "SUPERTREND_7_3": -0.2588, 
    "ADX_14": -0.5851, "COPPOCK": -0.2514, "ATR_14": 0.0899, 
    "VORTEX_14": 0.2762, "NVI": 0.0227, "DEMA_20": -0.2461, 
    "AD": 0.1758, "BBANDS_20_2": 0.8088, "EFI_20": -0.1310, 
    "ZSCORE_20": 0.5907, "NATR_20": 0.6832, "CMF_21": -0.3595, 
    "STOCH_21_5": 0.6101, "SMA_200": 0.1299, "HMA_9": -0.8518, 
    "AO_5_34": -0.5268, "LINREG_SLOPE_14": -0.4895, "BBANDS_10_1.5": 0.1388, 
    "PIVOTS": 0.6279, "KC_20_1.5": 0.6187, "KAMA_10": -0.0264, 
    "OBV": -0.0893
}


# ============================================================================
# STRATEGY 3: AGGRESSIVE (Maximum Profit)
# ============================================================================
# Performance: 51% Win Rate, $221k profit, highest profit
# Best for: Aggressive traders, larger accounts, high risk tolerance
AGGRESSIVE_WEIGHTS = {
    "PSAR": 1.0, "UO_7_14_28": 1.0, "NVI": 0.9591, 
    "ADX_20": -0.9140, "COPPOCK": -0.8994, "LINREG_SLOPE_14": -0.8898, 
    "AROON_25": 0.8784, "T3_10": -0.8648, "SMA_100": -0.7469, 
    "NATR_20": 0.7236, "MFI_14": -0.7001, "PIVOTS": -0.6529, 
    "WILLR_14": 0.6319, "AO_5_34": 0.6209, "ZSCORE_20": 0.6194, 
    "ATR_20": -0.5389, "TSI_13_25": 0.5093, "ADX_14": -0.4963, 
    "SUPERTREND_20_3": -0.4921, "OBV": 0.4714, "NATR_14": -0.3812, 
    "DEMA_10": 0.3766, "WILLR_28": 0.3677, "BBANDS_20_2.5": -0.3638, 
    "STOCH_14_3": 0.3542, "TEMA_10": 0.3504, "CMO_20": -0.3463, 
    "ICHIMOKU": 0.3182, "LINREG_SLOPE_25": -0.3069, "SUPERTREND_10_2": -0.2992, 
    "KC_20_1.5": 0.2962, "MOM_10": -0.2579, "MFI_20": -0.2482, 
    "CCI_20": -0.2279, "AROON_14": 0.2091, "KAMA_10": 0.2059, 
    "CMO_14": -0.1975, "STOCH_21_5": 0.1872, "KC_20_2": -0.1821, 
    "SUPERTREND_7_3": -0.1798, "TEMA_20": -0.1666, "HMA_16": -0.1209, 
    "CMF_20": -0.1076, "MASS_INDEX": -0.0946, "T3_5": 0.0679
}


# ============================================================================
# STRATEGY 4: OPTIMIZED (From Grid Search)
# ============================================================================
# Performance: Best overall from parameter optimization
# Best for: General purpose, validated parameters
OPTIMIZED_WEIGHTS = {
    "MFI_14": -0.8525, "TEMA_10": -0.3098, "SMA_20": 0.1499, 
    "AO_5_34": -0.9341, "ATR_14": 0.5290, "NATR_20": 0.2629, 
    "CMF_21": 0.3633, "TEMA_20": 0.4261, "TSI_13_25": 0.8902, 
    "STOCH_5_3": 0.6448, "LINREG_SLOPE_14": 0.1537, "AROON_25": -0.9552, 
    "CCI_20": -0.6396, "EFI_13": -0.8698, "VWMA_10": -0.9412, 
    "PIVOTS": -0.5856, "DONCHIAN_50": -0.3697, "BBANDS_20_2.5": 0.4697, 
    "WMA_20": -0.7451, "STOCH_14_3": -0.6456, "DEMA_20": -0.6598, 
    "VWMA_20": -0.8193, "ADOSC_3_10": 0.3516, "WMA_10": -0.7413, 
    "ADX_20": -0.8947, "ZSCORE_20": 0.5006, "ATR_20": 0.3462, 
    "UO_7_14_28": -0.2402, "VORTEX_14": -0.2020, "OBV": -0.4973, 
    "WILLR_14": -0.1467, "CCI_14": -0.3218, "KAMA_10": -0.1284, 
    "KST": -0.7961, "SUPERTREND_7_3": -0.9663, "MASS_INDEX": 0.4535, 
    "LINREG_SLOPE_25": -0.3096, "AROON_14": 0.4701, "MFI_20": -0.2728, 
    "PVI": 0.7962, "NVI": 0.8593, "ICHIMOKU": -0.4294, 
    "DEMA_10": -0.1344, "MOM_20": 0.2047, "PSAR": 0.1370, 
    "EFI_20": 0.1479, "DONCHIAN_20": 0.1299, "ROC_20": -0.0827
}


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    
    name: str
    description: str
    weights: Dict[str, float]
    dna_id: str
    
    # Trading parameters
    entry_threshold: float = 0.70
    exit_threshold: float = 0.30
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 0.0  # 0 = disabled
    
    # Risk parameters
    position_size_pct: float = 0.10
    max_position_value: float = 500_000.0
    max_positions: int = 5
    max_trades_per_day: int = 10
    daily_loss_limit_pct: float = 0.02
    
    # Expected performance (from backtests)
    expected_win_rate: float = 0.50
    expected_profit_factor: float = 1.0
    expected_sharpe: float = 0.0


# Pre-defined strategy configurations
STRATEGIES = {
    "SAFE": StrategyConfig(
        name="SAFE",
        description="High Win Rate Conservative (82% WR)",
        weights=SAFE_WEIGHTS,
        dna_id="dba0282a",
        entry_threshold=0.70,
        exit_threshold=0.30,
        stop_loss_atr_mult=2.0,
        position_size_pct=0.08,  # Conservative sizing
        expected_win_rate=0.82,
        expected_profit_factor=2.5,
    ),
    
    "BALANCED": StrategyConfig(
        name="BALANCED",
        description="Balanced Profit + Win Rate (61% WR, $136k)",
        weights=BALANCED_WEIGHTS,
        dna_id="0c53ead1",
        entry_threshold=0.70,
        exit_threshold=0.30,
        stop_loss_atr_mult=2.0,
        position_size_pct=0.10,
        expected_win_rate=0.61,
        expected_profit_factor=2.0,
        expected_sharpe=1.5,
    ),
    
    "AGGRESSIVE": StrategyConfig(
        name="AGGRESSIVE",
        description="Maximum Profit Strategy (51% WR, $221k)",
        weights=AGGRESSIVE_WEIGHTS,
        dna_id="178af481",
        entry_threshold=0.70,
        exit_threshold=0.30,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=4.0,  # Use take profits
        position_size_pct=0.15,  # Larger positions
        expected_win_rate=0.51,
        expected_profit_factor=1.8,
    ),
    
    "OPTIMIZED": StrategyConfig(
        name="OPTIMIZED",
        description="Grid Search Optimized (Best Overall)",
        weights=OPTIMIZED_WEIGHTS,
        dna_id="OPTIMIZED",
        entry_threshold=0.70,
        exit_threshold=0.30,
        stop_loss_atr_mult=2.0,
        position_size_pct=0.10,
    ),

    "PLAYER_1": StrategyConfig(
        name="PLAYER_1",
        description="Aggressive Growth ($221k profit, 50.6% WR)",
        weights={
            "PSAR": 1.0, "UO_7_14_28": 1.0, "NVI": 0.9591, "AROON_25": 0.8784,
            "NATR_20": 0.7236, "WILLR_14": 0.6319, "AO_5_34": 0.6209, "ZSCORE_20": 0.6194,
            "TSI_13_25": 0.5093, "OBV": 0.4714, "MFI_20": 0.4112, "ZSCORE_50": 0.3574,
            "ADX_20": -0.9140, "COPPOCK": -0.8994, "LINREG_SLOPE_14": -0.8898,
            "T3_10": -0.8648, "SMA_100": -0.7469, "MFI_14": -0.7001, "PIVOTS": -0.6529,
            "ATR_20": -0.5389, "ADX_14": -0.4963, "SUPERTREND_20_3": -0.4921, "CMF_21": -0.4400,
        },
        dna_id="178af481",
        entry_threshold=0.70,
        exit_threshold=0.30,
    ),

    "PLAYER_2": StrategyConfig(
        name="PLAYER_2",
        description="High Precision ($55k profit, 82.4% WR)",
        weights={
            "TEMA_20": 0.9034, "KST": 0.8221, "CMF_20": 0.7471, "HMA_9": 0.6080,
            "CMO_14": 0.5187, "ZSCORE_20": 0.4706, "BBANDS_20_2": 0.3865, "VORTEX_14": 0.3862,
            "SMA_100": 0.3522, "ZSCORE_50": 0.3520, "WMA_20": 0.3320, "VWMA_10": 0.2957,
            "ADX_20": -1.0, "DONCHIAN_20": -0.6885, "SUPERTREND_7_3": -0.5066, "PVI": -0.4915,
            "TRUERANGE": -0.4556, "ADX_14": -0.4027, "LINREG_SLOPE_14": -0.3821, "OBV": -0.3075,
        },
        dna_id="3d290598",
        entry_threshold=0.70,
        exit_threshold=0.30,
    ),

    "PLAYER_3": StrategyConfig(
        name="PLAYER_3",
        description="Balanced Performer ($89k profit, 78.3% WR)",
        weights={
            "TSI_13_25": 0.8883, "NVI": 0.8611, "PVI": 0.7964, "STOCH_5_3": 0.6260,
            "ATR_14": 0.5265, "ZSCORE_20": 0.5002, "AROON_14": 0.4698, "BBANDS_20_2.5": 0.4642,
            "MASS_INDEX": 0.4535, "TEMA_20": 0.4258, "CMF_21": 0.3670, "ATR_20": 0.3443,
            "SUPERTREND_7_3": -0.9663, "AROON_25": -0.9508, "AO_5_34": -0.9286,
            "VWMA_10": -0.9283, "ADX_20": -0.8928, "EFI_13": -0.8716, "MFI_14": -0.8352,
            "VWMA_20": -0.8194, "KST": -0.7956, "WMA_20": -0.7559, "WMA_10": -0.7482,
        },
        dna_id="8748f3f8",
        entry_threshold=0.70,
        exit_threshold=0.30,
    ),

    "PLAYER_4": StrategyConfig(
        name="PLAYER_4",
        description="Trend Follower ($207k profit, 40.6% WR)",
        weights={
            "PSAR": 1.0, "NVI": 0.9591, "LINREG_SLOPE_25": 0.8524, "AROON_25": 0.8297,
            "NATR_20": 0.7236, "UO_7_14_28": 0.6906, "WILLR_14": 0.6319, "ZSCORE_20": 0.6194,
            "TSI_13_25": 0.5093, "OBV": 0.4714, "MFI_20": 0.4112, "ZSCORE_50": 0.3574,
            "COPPOCK": -0.8994, "T3_10": -0.8648, "LINREG_SLOPE_14": -0.8423, "SMA_100": -0.7469,
            "MFI_14": -0.7001, "ADX_20": -0.6628, "PIVOTS": -0.6529, "ATR_20": -0.5389,
        },
        dna_id="7d8d1a12",
        entry_threshold=0.70,
        exit_threshold=0.30,
    ),

    "PLAYER_5": StrategyConfig(
        name="PLAYER_5",
        description="Robust Hybrid ($178k profit, 54.5% WR)",
        weights={
            "NATR_20": 0.9810, "NVI": 0.9591, "PSAR": 0.8580, "VWMA_20": 0.7125,
            "UO_7_14_28": 0.6423, "ZSCORE_20": 0.6375, "WILLR_14": 0.6319, "AROON_25": 0.5984,
            "TSI_13_25": 0.5022, "OBV": 0.4714, "ZSCORE_50": 0.4004, "LINREG_SLOPE_25": 0.3927,
            "LINREG_SLOPE_14": -0.9212, "COPPOCK": -0.8994, "EFI_20": -0.8544, "SMA_100": -0.7469,
            "ADX_20": -0.6926, "PIVOTS": -0.6529, "ATR_20": -0.5389, "ADX_14": -0.4963,
        },
        dna_id="321c0c6f",
        entry_threshold=0.70,
        exit_threshold=0.30,
    ),
}


def get_strategy(name: str) -> StrategyConfig:
    """Get a strategy by name."""
    name = name.upper()
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return STRATEGIES[name]


def get_default_strategy() -> StrategyConfig:
    """Get the default strategy (BALANCED)."""
    return STRATEGIES["BALANCED"]


def list_strategies() -> Dict[str, str]:
    """List all available strategies with descriptions."""
    return {name: s.description for name, s in STRATEGIES.items()}
