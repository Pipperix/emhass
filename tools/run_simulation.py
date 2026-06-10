"""
Entry-point for the EMHASS Day-Ahead + MPC simulation.

Usage:
    python run_simulation.py [--config CONFIG_FILE] [--dataset DATASET_FILE]

Arguments:
    --config   Name of the config file located in the 'configs/' folder.
               (default: dynamic_miner_config.json)
               You can pass it with or without the '.json' extension.
               
    --dataset  Name of the dataset file located in the 'datasets/' folder.
               (default: scenario_2026_05_09.csv)
               You can pass it with or without the '.csv' extension.

Examples:
    # Run with default configuration and dataset
    python run_simulation.py

    # Run with a specific config and dataset
    python run_simulation.py --config custom_config --dataset scenario_D1
"""

import json
import os
import pathlib
import sys
import argparse

# Configure HIGHS solver for CVXPY
os.environ["LP_SOLVER"] = "HIGHS"

current_dir = pathlib.Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from logic import load_and_prepare_dataset, EmhassSimulator, generate_mpc_plot, generate_dayahead_plot


def main():
    parser = argparse.ArgumentParser(description="EMHASS Simulation Runner")
    parser.add_argument("--config", type=str, default="dynamic_miner_config.json", help="Name of the config file (inside configs/)")
    parser.add_argument("--dataset", type=str, default="scenario_2026_05_09.csv", help="Name of the dataset file (inside datasets/)")
    args = parser.parse_args()

    base_dir = current_dir

    # Ensure extensions are present
    config_name = args.config if args.config.endswith(".json") else f"{args.config}.json"
    dataset_name = args.dataset if args.dataset.endswith(".csv") else f"{args.dataset}.csv"

    config_path = base_dir / "configs" / config_name
    dataset_path = base_dir / "datasets" / dataset_name

    config_base = config_name.replace('.json', '')
    dataset_base = dataset_name.replace('.csv', '')
    combo_name = f"{config_base}_{dataset_base}"

    out_dir = base_dir / "output" / combo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv_dayahead = out_dir / "dayahead_result.csv"
    out_csv_mpc = out_dir / "optimization_result.csv"
    out_html_mpc = out_dir / "optimization_result.html"
    out_html_dayahead = out_dir / "dayahead_result.html"
    out_summary = out_dir / "summary.txt"

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
        # IMPORTANT: Re-initialize the simulator to clear the cache and the optimizer state
        # This prevents the C++ crash (malloc error) caused by Day-Ahead mutating num_timesteps
        simulator_mpc = EmhassSimulator(config)
        df_mpc = simulator_mpc.simulate_mpc(df_data)
        df_mpc.to_csv(out_csv_mpc)

        # === 6. Visualization ===
        print("6. Generating final Plotly chart...")
        generate_mpc_plot(str(out_csv_mpc), str(out_html_mpc))

        # === 7. Summary ===
        time_step_min = config["time_step_min"]
        
        def_load_names = [d["name"] for d in config["deferrable_loads"]]
        
        # Energy metrics (kWh)
        total_pv_production_kwh = (df_mpc["pv_actual"].sum() * time_step_min / 60) / 1000
        total_grid_import_kwh = (df_mpc["grid_power"].clip(lower=0).sum() * time_step_min / 60) / 1000
        total_grid_export_kwh = (df_mpc["grid_power"].clip(upper=0).abs().sum() * time_step_min / 60) / 1000
        total_house_load_kwh = (df_mpc["load_actual"].sum() * time_step_min / 60) / 1000
        total_appliance_energy_kwh = (df_mpc[def_load_names].sum().sum() * time_step_min / 60) / 1000
        total_consumption_kwh = total_house_load_kwh + total_appliance_energy_kwh
        
        # Appliance specific energy
        appliance_energy_dict = {
            name: (df_mpc[name].sum() * time_step_min / 60) / 1000 
            for name in def_load_names
        }

        # Key indicators
        self_sufficiency = 1 - (total_grid_import_kwh / total_consumption_kwh) if total_consumption_kwh > 0 else 1.0

        summary_lines = [
            "--- Simulation Summary ---",
            f"Total PV Production:     {total_pv_production_kwh:.2f} kWh",
            f"Total Grid Import:       {total_grid_import_kwh:.2f} kWh",
            f"Total Grid Export:       {total_grid_export_kwh:.2f} kWh",
            f"Total Consumption:       {total_consumption_kwh:.2f} kWh",
            f"  - House Load:          {total_house_load_kwh:.2f} kWh"
        ]
        for name, energy in appliance_energy_dict.items():
            summary_lines.append(f"  - Appliance ({name}): {energy:.2f} kWh")
        summary_lines.append(f"Self-Sufficiency:        {self_sufficiency:.2%}")
        summary_lines.append(f"Final Battery SOC:       {df_mpc['batt_soc'].iloc[-1]:.1%}")

        summary_text = "\n".join(summary_lines)
        print(f"\n{summary_text}")

        with open(out_summary, "w") as f:
            f.write(summary_text + "\n")

    except Exception as e:
        print(f"   Error in MPC execution: {e}")

    print("\n[OK] Simulation pipeline completed.")


if __name__ == "__main__":
    main()
