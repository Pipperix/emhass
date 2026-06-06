"""
Entry-point for the EMHASS Day-Ahead + MPC simulation.
"""

import json
import os
import pathlib
import sys

# Configure HIGHS solver for CVXPY
os.environ["LP_SOLVER"] = "HIGHS"

current_dir = pathlib.Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from logic import load_and_prepare_dataset, EmhassSimulator, generate_mpc_plot, generate_dayahead_plot


def main():
    base_dir = current_dir

    config_path = base_dir / "configs" / "no_var_loads.json"
    dataset_path = base_dir / "datasets" / "scenario_2026_05_09.csv"

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
        str(dataset_path), time_step_min=config["time_step_min"]
    )

    # === 3. Initialize EmhassSimulator ===
    print("3. Initializing EmhassSimulator (fail-fast mode)...")
    simulator = EmhassSimulator(config)

    # === 4. Day-Ahead Optimization ===
    print("4. Running Day-Ahead Optimization...")
    try:
        df_dayahead = simulator.simulate_day_ahead(df_data)
        df_dayahead.to_csv(out_csv_dayahead)
        generate_dayahead_plot(str(out_csv_dayahead), str(out_html_dayahead), config)
        print(f"   Day-Ahead completed.")
    except Exception as e:
        print(f"   Error in Day-Ahead: {e}")

    # === 5. MPC Closed-Loop ===
    print("5. Running MPC Closed-Loop simulation...")
    try:
        df_mpc = simulator.simulate_mpc(df_data)
        df_mpc.to_csv(out_csv_mpc)

        # === 6. Visualization ===
        print("6. Generating final Plotly chart...")
        generate_mpc_plot(str(out_csv_mpc), str(out_html_mpc))

        # === 7. Summary ===
        time_step_min = config["time_step_min"]
        
        # Calculate total energy for ALL deferrable loads
        def_load_names = [d["name"] for d in config["deferrable_loads"]]
        total_appliance_energy_kwh = (
            df_mpc[def_load_names].sum().sum() * time_step_min / 60
        ) / 1000
        
        total_grid_import_kwh = (
            df_mpc["grid_power"].clip(lower=0).sum() * time_step_min / 60
        ) / 1000
        total_load_kwh = (
            df_mpc["load_actual"].sum() * time_step_min / 60
        ) / 1000

        print("\n--- Simulation Summary ---")
        print(f"Appliances Configured:   {len(def_load_names)} ({', '.join(def_load_names)})")
        print(f"Total Appliances Energy: {total_appliance_energy_kwh:.2f} kWh")
        print(f"Total Grid Import:       {total_grid_import_kwh:.2f} kWh")
        print(f"Self-Consumption Ratio:  {(1 - total_grid_import_kwh / (total_load_kwh + total_appliance_energy_kwh)):.2%}")
        print(f"Final Battery SOC:       {df_mpc['batt_soc'].iloc[-1]:.1%}")

    except Exception as e:
        print(f"   Error in MPC execution: {e}")

    print("\n[OK] Simulation pipeline completed.")


if __name__ == "__main__":
    main()
