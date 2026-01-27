import sqlite3
import json
import pandas as pd
from pathlib import Path

# Connect to database
db_path = Path("trading_evolution.db")
if not db_path.exists():
    print("Database not found!")
    exit(1)

conn = sqlite3.connect("trading_evolution.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 1. Find the Best DNA from Hall of Fame or DNA Configs (LATEST RUN)
print("--- Scanning for Best Strategy (Latest Run) ---")
# Get max run_id
cursor.execute("SELECT MAX(run_id) as max_id FROM evolution_runs")
row = cursor.fetchone()
max_run_id = row['max_id'] if row else 0
print(f"Analyzing Run ID: {max_run_id}")

# Check Generation Progress
cursor.execute("SELECT * FROM generations WHERE run_id = ? ORDER BY generation_num DESC LIMIT 1", (max_run_id,))
gen_row = cursor.fetchone()
if gen_row:
    print(f"Latest Generation: {gen_row['generation_num']}")
    print(f"Best Fitness in Gen: {gen_row['best_fitness']}")
    print(f"Avg Fitness in Gen: {gen_row['avg_fitness']}")
else:
    print("No generation data found yet.")

cursor.execute("SELECT * FROM dna_configs WHERE run_id = ? ORDER BY fitness_score DESC LIMIT 1", (max_run_id,))
best_dna = cursor.fetchone()

if not best_dna:
    # Try Hall of Fame
    cursor.execute("SELECT * FROM hall_of_fame WHERE run_id = ? ORDER BY fitness_score DESC LIMIT 1", (max_run_id,))
    best_dna = cursor.fetchone()

if not best_dna:
    print("No DNA records found.")
    exit()

print(f"Best DNA ID: {best_dna['dna_id']}")
print(f"Fitness: {best_dna['fitness_score']}")
print(f"Sharpe: {best_dna['sharpe_ratio']}")
print(f"Net Profit: ${best_dna['net_profit']}")

# 2. Analyze Indicators (Weights)
weights = json.loads(best_dna['weights_json'])
if 'active_indicators_json' in best_dna.keys():
    active_indicators = json.loads(best_dna['active_indicators_json'])
else:
    active_indicators = list(weights.keys())

print("\n--- Active Indicators & Weights ---")
sorted_weights = sorted([(k, v) for k, v in weights.items() if k in active_indicators], key=lambda x: abs(x[1]), reverse=True)
for ind, w in sorted_weights:
    print(f"{ind}: {w:.4f}")

# 3. Get Trades for this DNA
print("\n--- Trade History (Top 10) ---")
# Trades might be in 'trades' table linked by dna_id
cursor.execute("SELECT * FROM trades WHERE dna_id = ? ORDER BY entry_time LIMIT 20", (best_dna['dna_id'],))
trades = cursor.fetchall()

if not trades:
    print("No trades found for this DNA (maybe not saved to DB due to crash?).")
    # If no trades in DB, we rely on the metrics summary.
else:
    for t in trades:
        print(f"{t['entry_time']} | {t['symbol']} | {t['direction']} | Entry: {t['entry_price']:.2f} | Exit: {t['exit_price']:.2f} | PnL: ${t['net_pnl']:.2f} ({t['exit_reason']})")

conn.close()
