#!/usr/bin/env python3
"""
Run 50-run simulation for all 5 Players.
"""
import subprocess
import time
import json
import pandas as pd
from pathlib import Path

PLAYERS = ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]

def run_simulation(player_name, runs=50):
    print(f"\n{'='*60}")
    print(f"STARTING SIMULATION FOR {player_name} ({runs} runs)")
    print(f"{'='*60}")
    
    cmd = [
        "python3", "run_trading_system.py",
        "--backtest",
        "--strategy", player_name,
        "--iterations", str(runs),
        "--years", "1"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {player_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {player_name} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user.")
        return False
        
    return True

def export_to_excel():
    print("\n📊 Aggregating results to Excel...")
    results_dir = Path("trading_results")
    all_data = []

    # Find the most recent run file for each player
    # Since we can't easily map process to file without parsing stdout, 
    # we'll look for files matching our naming convention created today.
    # Note: This simply grabs ALL JSONs in the folder and filters for expected keys.
    # Ideally we'd be more rigorous, but this works for a fresh batch.
    
    json_files = sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    processed_strategies = set()
    
    for f in json_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                
            sid = data.get("strategy_id")
            # Only process each strategy once (the latest run)
            # We need to map strategy_id back to PLAYER name if possible, 
            # or rely on the fact we just ran them.
            # Strategy ID mapping from strategies.py:
            # P1=178af481, P2=3d290598, P3=8748f3f8, P4=7d8d1a12, P5=321c0c6f
            
            # Or simpler: Just include all found valid result files for today
            if "iteration_history" in data:
                # Add rows for every iteration
                for entry in data["iteration_history"]:
                    metrics = entry["metrics"]
                    row = {
                        "File": f.name,
                        "Strategy_ID": sid,
                        "Version": entry["strategy_version"],
                        "Run_Iteration": entry["iteration"],
                        "P&L": metrics.get("pnl"),
                        "Win_Rate": metrics.get("win_rate"),
                        "Sharpe": metrics.get("sharpe"),
                        "Trades": metrics.get("trades"),
                        "Drawdown": metrics.get("drawdown"),
                        "Timestamp": data.get("timestamp")
                    }
                    all_data.append(row)
                    
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        outfile = "simulation_results_50_runs.xlsx"
        df.to_excel(outfile, index=False)
        print(f"✅ Excel report saved to {outfile}")
        print(f"Total rows: {len(df)}")
    else:
        print("❌ No data found to export.")

if __name__ == "__main__":
    success = True
    for player in PLAYERS:
        if not run_simulation(player, runs=50):
            success = False
            break
    
    if success:
        export_to_excel()
        print("\nALL SIMULATIONS COMPLETED & EXPORTED.")

