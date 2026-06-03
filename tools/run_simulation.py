"""
Entry-point for the EMHASS Day-Ahead + MPC simulation.

Loads configuration from JSON, prepares the dataset (with forecast/actual noise),
runs the Day-Ahead open-loop optimization first, then the MPC closed-loop,
saving results to CSV and generating Plotly visualizations.
"""

import json
import os
import pathlib
import sys

# Configure HIGHS solver for CVXPY (consistent with the original benchmark)
os.environ["LP_SOLVER"] = "HIGHS"

# Add tools folder to path for easy 'logic' imports
current_dir = pathlib.Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from logic import load_and_prepare_dataset, EmhassSimulator, generate_mpc_plot, generate_dayahead_plot


def main():
    base_dir = current_dir

    config_path = base_dir / "configs" / "sim_config.json"
    dataset_path = base_dir / "datasets" / "scenario_2026_05_09.csv"

    # Output paths
    out_csv_dayahead = base_dir / "output" / "benchmark_results_dayahead.csv"
    out_csv_mpc = base_dir / "output" / "benchmark_results_simulation.csv"
    out_html_mpc = base_dir / "output" / "benchmark_visualization.html"
    out_html_dayahead = base_dir / "output" / "benchmark_visualization_dayahead.html"

    # === 1. Load configuration ===
    print("1. Loading configuration...")
    with open(config_path, "r") as f:
        config = json.load(f)

    # === 2. Load and prepare dataset ===
    print(f"2. Loading and preparing dataset ({dataset_path.name})...")
    df_data = load_and_prepare_dataset(
        str(dataset_path), time_step_min=config.get("time_step_min", 30)
    )
    print(f"   Found {len(df_data)} timesteps to simulate.")

    # === 3. Initialize EmhassSimulator ===
    print("3. Initializing EmhassSimulator (native EMHASS pipeline)...")
    simulator = EmhassSimulator(config)

    # === 4. Day-Ahead Optimization ===
    print("4. Running Day-Ahead Optimization (open-loop)...")
    try:
        df_dayahead = simulator.simulate_day_ahead(df_data)
        df_dayahead.to_csv(out_csv_dayahead)
        print(f"   Day-Ahead completed. Saved: {out_csv_dayahead.name}")
        generate_dayahead_plot(str(out_csv_dayahead), str(out_html_dayahead))
    except Exception as e:
        print(f"   Error in Day-Ahead: {e}")
        import traceback
        traceback.print_exc()

    # === 5. MPC Closed-Loop ===
    print("5. Running MPC Closed-Loop simulation...")
    try:
        df_mpc = simulator.simulate_mpc(df_data)
        df_mpc.to_csv(out_csv_mpc)
        print(f"   MPC completed. Saved: {out_csv_mpc.name}")

        # === 6. Visualization ===
        print("6. Generating final Plotly chart...")
        generate_mpc_plot(str(out_csv_mpc), str(out_html_mpc))

        # === 7. Summary ===
        time_step_min = config.get("time_step_min", 30)
        total_miner_energy_kwh = (
            df_mpc["miner_power"].sum() * time_step_min / 60
        ) / 1000
        total_grid_import_kwh = (
            df_mpc["grid_power"].clip(lower=0).sum() * time_step_min / 60
        ) / 1000
        total_load_kwh = (
            df_mpc["load_actual"].sum() * time_step_min / 60
        ) / 1000

        print("\n--- Benchmark Summary ---")
        print(f"Total Miner Consumption: {total_miner_energy_kwh:.2f} kWh")
        print(f"Total Grid Import:       {total_grid_import_kwh:.2f} kWh")
        print(
            f"Self-Consumption Ratio:  "
            f"{(1 - total_grid_import_kwh / (total_load_kwh + total_miner_energy_kwh)):.2%}"
        )
        print(f"Final Battery SOC:       {df_mpc['batt_soc'].iloc[-1]:.1%}")

    except Exception as e:
        print(f"   Error in MPC execution or Plot: {e}")
        import traceback
        traceback.print_exc()

    print("\n[OK] Simulation pipeline completed.")


if __name__ == "__main__":
    main()
