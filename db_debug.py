
import sqlite3
import pandas as pd

def inspect_runs():
    conn = sqlite3.connect('trading_evolution.db')
    try:
        # Get all runs
        runs = pd.read_sql_query("SELECT * FROM evolution_runs ORDER BY run_id DESC LIMIT 10", conn)
        print("Recent Runs:")
        print(runs[['run_id', 'started_at', 'total_generations', 'final_fitness', 'status']])

        print("\nGeneration Counts per Run:")
        for run_id in runs['run_id']:
            gen_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM generations WHERE run_id = {run_id}", conn).iloc[0]['count']
            dna_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM dna_registry WHERE run_id = {run_id}", conn).iloc[0]['count']
            hof_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM hall_of_fame WHERE run_id = {run_id}", conn).iloc[0]['count']
            print(f"Run ID {run_id}: {gen_count} generations, {dna_count} DNAs, {hof_count} HoF entries")

    finally:
        conn.close()

if __name__ == "__main__":
    inspect_runs()
