#!/usr/bin/env python3
"""
THE ULTIMATE EVOLUTION: 10 EPOCHS x 2 YEARS x WEEKLY COACH
==========================================================

HIERARCHY:
1. MACRO-EVOLUTION (Architect): Runs 10 'Epochs'. Before each Epoch, actively redesigns the 
   Indicator DNA (choosing from 130+ indicators) based on previous Epoch's multi-year performance.

2. MICRO-EVOLUTION (Coach): During each 2-Year Epoch, monitors the Player DAILY/WEEKLY.
   Adjusts parameters (thresholds) dynamically to fit market regimes.

3. EXECUTION (Player): Trades autonomously using the current DNA + Coach's parameter adjustments.

This combines:
- Deep Library Access (130+ Indicators)
- Long-Term Stability (2-Year Backtests)
- Fast Adaptation (Weekly Feedback)
- Iterative Learning (10 Generations)
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from dotenv import load_dotenv
import google.generativeai as genai

# Add project root to path
sys.path.insert(0, '/Users/krishjain/Desktop/New Project')

# Import Internal Modules
from trading_evolution.indicators.universe import IndicatorUniverse
from trading_evolution.indicators.calculator import IndicatorCalculator
from trading_evolution.indicators.normalizer import IndicatorNormalizer
from trading_evolution.super_indicator.core import SuperIndicator
from trading_evolution.super_indicator.dna import SuperIndicatorDNA, create_dna_from_weights

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    print("✅ Gemini API connected (gemini-3-flash-preview)")
else:
    print("❌ No API key found")
    model = None


# ============================================================================
# ARCHITECT (MACRO-EVOLUTION) - Between Epochs
# ============================================================================

class GrandArchitect:
    def __init__(self, universe: IndicatorUniverse):
        self.universe = universe
        self.available_indicators = universe.get_all()
        self.categories = universe.get_categories()
        self.memory = []

    def redesign_dna(self, epoch: int, previous_dna: SuperIndicatorDNA, performance: dict) -> SuperIndicatorDNA:
        """design a new DNA for the next 2-year run."""
        if not model: return previous_dna
        
        active_genes = {n: g.weight for n, g in previous_dna.genes.items() if g.active}
        
        prompt = f"""You are the GRAND ARCHITECT optimizing a trading system over 2-year cycles.
EPOCH: {epoch}/10

PREVIOUS EPOCH PERFORMANCE (2 Years):
- P&L: ₹{performance['pnl']:,.0f}
- Win Rate: {performance['win_rate']:.1f}%
- Trades: {performance['trades']}
- Sharpes: {performance.get('sharpe', 0.0):.2f}

PREVIOUS DNA:
{json.dumps(active_genes, indent=2)}

AVAILABLE LIBRARY: {len(self.available_indicators)} Indicators across {', '.join(self.categories)}.

TASK:
Based on the 2-year results, RE-ENGINEER the indicator mix for the next Epoch.
- If trades == 0: You MUST pick more sensitive indicators (Momentum/Stochastics).
- If P&L < 0: Remove noise, add Trend confirmation (ADX, SuperTrend).
- If Win Rate < 50%: Add mean-reversion filters (Bollinger, RSI).

Respond in JSON:
{{
  "diagnosis": "Analysis of why previous epoch failed/succeeded",
  "blueprint": {{
    "RSI_14": 0.4,
    "MACD_12_26_9": 0.3,
    "ADX_14": 0.2,
    "ATR_14": -0.1
  }}
}}
"""
        try:
            time.sleep(2.0)
            response = model.generate_content(prompt)
            data = json.loads(response.text[response.text.find('{'):response.text.rfind('}')+1])
            print(f"\n🏛️ ARCHITECT (Epoch {epoch}): {data.get('diagnosis', '')}")
            
            new_weights = data.get('blueprint', {})
            cats = {}
            for name in new_weights:
                defn = self.universe.get_definition(name)
                if defn: cats[name] = defn.category
                
            return create_dna_from_weights(new_weights, categories=cats, generation=epoch)
            
        except Exception as e:
            print(f"  ⚠️ Architect Error: {e}")
            return previous_dna


# ============================================================================
# COACH (MICRO-EVOLUTION) - Weekly during Run
# ============================================================================

class WeeklyCoach:
    def __init__(self):
        self.logs = []
        
    def analyze_week(self, trades, dna_config, date_str):
        """Review weekly performance and tweak thresholds."""
        if not model: return {}
        
        # Simple heuristic fallback if no trades
        if not trades:
            return {'entry_delta': -0.02, 'exit_delta': 0.0, 'reason': 'Force participation'}
            
        pnl = sum(t['pnl'] for t in trades)
        
        prompt = f"""You are a Tactical Trading Coach.
DATE: {date_str}
WEEKLY UPDATE: {len(trades)} trades, P&L ₹{pnl:.0f}
CURRENT PARAMS: Entry > {dna_config['entry']:.3f}, Exit < {dna_config['exit']:.3f}

DECISISON:
Adjust thresholds slightly (-0.03 to +0.03) to improve next week.
- Raise Entry if too many false signals.
- Lower Entry if missing trades.

JSON Response: {{ "entry_delta": 0.0, "exit_delta": 0.0, "reason": "..." }}
"""
        try:
            # Fast/Cheap call (simulated or real) -> Using heuristic to save API calls for Architect
            # Actually, let's just use simple logic here to speed up the massive simulation
            # The Architect does the heavy lifting. The Coach here will be a rule-based expert system 
            # to avoid 1000s of API calls in one run.
            
            if pnl > 0:
                return {'entry_delta': 0.0, 'exit_delta': 0.01, 'reason': 'Profitable, hold longer'}
            else:
                return {'entry_delta': 0.01, 'exit_delta': -0.01, 'reason': 'Losses, tighten filters'}
                
        except: return {}


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

def load_environment():
    print("📊 Loading 2-Year Data Environment (Jan 2024 - Jan 2026) for 30 Symbols...")
    # 30 Major Symbols (Sensex 30 Proxy)
    symbols = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'HCLTECH.NS', 'MARUTI.NS', 
        'TITAN.NS', 'BAJFINANCE.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'ULTRACEMCO.NS', 
        'NESTLEIND.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATASTEEL.NS', 
        'JSWSTEEL.NS', 'ADANIENT.NS', 'ADANIPORTS.NS', 'COALINDIA.NS', 'WIPRO.NS'
    ]
    
    universe = IndicatorUniverse()
    universe.load_all()
    calculator = IndicatorCalculator(universe)
    normalizer = IndicatorNormalizer()
    
    env_data = {}
    
    for sym in symbols:
        try:
            # Get 2 years of data
            df = yf.Ticker(sym).history(start="2024-01-01", end="2026-01-01", interval="1d")
            if len(df) < 200: continue
            
            # Calculate ALL 130+ indicators once
            raw = calculator.calculate_all(df)
            raw = calculator.rename_to_dna_names(raw)
            norm = normalizer.normalize_all(raw, price_series=df['Close'])
            
            env_data[sym] = {'price': df, 'indicators': norm}
            print(f"   ✅ {sym}: Ready ({len(df)} days)")
        except: pass
        
    return env_data, universe

def run_ultimate_evolution():
    print("=" * 80)
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║   THE ULTIMATE EVOLVER: 10 EPOCHS x 2 YEARS x WEEKLY FEEDBACK                ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    # 1. Load Enviroment
    env_data, universe = load_environment()
    architect = GrandArchitect(universe)
    coach = WeeklyCoach()
    
    # 2. Initial DNA — start from best evolved strategy (PLAYER_3 / DNA 8748f3f8)
    #    instead of a single indicator, so we begin with proven Sharpe/WR.
    from trading_evolution.paper.paper_trader import BEST_STRATEGY
    dna = create_dna_from_weights(BEST_STRATEGY['weights'], generation=0)
    
    # 3. Macro-Evolution Loop (10 Epochs)
    global_history = []
    
    for epoch in range(1, 51):
        print(f"\n🌍 EPOCH {epoch}/10: Running 2-Year Simulation...")
        print(f"   🧬 DNA: {list(dna.genes.keys())}")
        
        si_engine = SuperIndicator(dna)
        
        # Simulation State
        total_pnl = 0
        all_trades = []
        
        # Dynamic Thresholds (reset each epoch)
        params = {'entry': 0.65, 'exit': 0.35}
        
        # Run 2-Year Timeline (approx 100 weeks)
        # We iterate by weeks for speed, treating daily data in chunks
        all_dates = sorted(env_data['RELIANCE.NS']['price'].index)
        weeks = [all_dates[i:i+5] for i in range(0, len(all_dates), 5)]
        
        for week_days in weeks:
            week_trades = []
            
            for sym, data in env_data.items():
                df = data['price']
                indicators = data['indicators']
                
                # Slicing is expensive, so we do index matching
                # Optimize: Just check if date in index
                valid_days = [d for d in week_days if d in df.index]
                if not valid_days: continue
                
                # Get signal for this week's days
                # Note: 'calculate' is vectorized, we should pre-calc SI for speed?
                # No, DNA changes between epochs, so we must calc per epoch.
                # But we can calc signal for whole dataframe once per epoch per symbol!
                pass # Logic moved outside loop for speed
            
        # -- FAST SIMULATION BLOCK --
        # To make 10x2years run in reasonable time:
        # 1. Calc SI for full 2 years for all symbols
        # 2. Iterate days
        
        epoch_si_map = {}
        for sym, data in env_data.items():
            epoch_si_map[sym] = si_engine.calculate(data['indicators']).fillna(0.5)
            
        # Timeline Loop
        for week_idx, week_days in enumerate(weeks):
            week_trades_list = []
            
            # Coach Logic (Weekly)
            if week_idx > 0 and week_idx % 4 == 0: # Monthly feedback to save time/logs
                advice = coach.analyze_week(all_trades[-5:], params, str(week_days[0].date()))
                if advice and advice.get('reason'):
                     params['entry'] = np.clip(params['entry'] + advice.get('entry_delta',0), 0.4, 0.9)
                     params['exit'] = np.clip(params['exit'] + advice.get('exit_delta',0), 0.1, 0.6)
            
            # Trading — track position state across weeks per symbol
            if week_idx == 0:
                # Initialize per-symbol position tracking at start of epoch
                if 'pos_state' not in dir():
                    pos_state = {}  # {sym: (in_pos, entry_price)}
                pos_state = {sym: (False, 0.0) for sym in env_data}

            for sym, data in env_data.items():
                si_series = epoch_si_map[sym]
                price_series = data['price']['Close']

                in_pos, entry_price = pos_state.get(sym, (False, 0.0))

                for day in week_days:
                    if day not in si_series.index: continue

                    val = si_series.loc[day]
                    price = price_series.loc[day]

                    if not in_pos and val > params['entry']:
                        in_pos = True
                        entry_price = price
                    elif in_pos and val < params['exit']:
                        pnl = price - entry_price
                        trade = {'pnl': pnl, 'date': day, 'symbol': sym}
                        week_trades_list.append(trade)
                        all_trades.append(trade)
                        total_pnl += pnl
                        in_pos = False

                pos_state[sym] = (in_pos, entry_price)
                        
        # End of Epoch
        wins = sum(1 for t in all_trades if t['pnl'] > 0)
        count = len(all_trades)
        win_rate = (wins/count*100) if count > 0 else 0
        
        print(f"   📊 Epoch Result: ₹{total_pnl:,.0f} | {count} Trades | WR: {win_rate:.1f}%")
        
        # Architect Redesign
        perf = {'pnl': total_pnl, 'trades': count, 'win_rate': win_rate}
        dna = architect.redesign_dna(epoch, dna, perf)
        global_history.append(perf)
        
    print("\n" + "="*80)
    print("✅ ULTIMATE EVOLUTION COMPLETE")
    print("Best Epoch P&L: ₹", max(p['pnl'] for p in global_history))

if __name__ == "__main__":
    run_ultimate_evolution()
