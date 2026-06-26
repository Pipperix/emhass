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
import pandas as pd

# Configure HIGHS solver for CVXPY
os.environ["LP_SOLVER"] = "HIGHS"

current_dir = pathlib.Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from logic import load_and_prepare_dataset, EmhassSimulator, generate_mpc_plot, generate_dayahead_plot, generate_comparison_plot


def run_single_scenario(config_name_raw, dataset_name_raw, base_dir):
    # Ensure extensions are present
    config_name = config_name_raw if config_name_raw.endswith(".json") else f"{config_name_raw}.json"
    dataset_name = dataset_name_raw if dataset_name_raw.endswith(".csv") else f"{dataset_name_raw}.csv"

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
    out_html_comparison = out_dir / "comparison_result.html"
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
    print("4. Running Day-Ahead Optimization (split by day)...")
    try:
        time_step_min = config["time_step_min"]
        steps_per_day = int(24 * 60 / time_step_min)
        
        # Day 1
        df_data_day1 = df_data.iloc[:steps_per_day]
        df_dayahead_day1 = simulator.simulate_day_ahead(df_data_day1)
        
        # Day 2
        original_initial_soc = simulator.config["initial_soc"]
        # Update SOC for Day 2 based on the end of Day 1
        simulator.config["initial_soc"] = df_dayahead_day1["SOC_opt"].iloc[-1]
        
        df_data_day2 = df_data.iloc[steps_per_day : 2 * steps_per_day]
        df_dayahead_day2 = simulator.simulate_day_ahead(df_data_day2)
        
        # Restore original SOC just in case it's used elsewhere
        simulator.config["initial_soc"] = original_initial_soc
        
        # Concatenate and save the full 48h Day-Ahead result
        df_dayahead = pd.concat([df_dayahead_day1, df_dayahead_day2])
        df_dayahead.to_csv(out_csv_dayahead)
        
        # Plot the Day-Ahead results (visualizer internally limits to 24h)
        generate_dayahead_plot(str(out_csv_dayahead), str(out_html_dayahead), config, config_base, dataset_base)
        
        print(f"   Day-Ahead completed. (Plotted Day 1 only)")
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
        print("6. Generating final Plotly charts...")
        generate_mpc_plot(str(out_csv_mpc), str(out_html_mpc), config_base, dataset_base)
        generate_comparison_plot(str(out_csv_mpc), str(out_csv_dayahead), str(out_html_comparison), config, config_base, dataset_base)

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
        self_consumption = 1 - (total_grid_export_kwh / total_pv_production_kwh) if total_pv_production_kwh > 0 else 1.0
        
        self_sufficiency = max(0.0, min(1.0, self_sufficiency))
        self_consumption = max(0.0, min(1.0, self_consumption))

        # Economic metrics
        total_purchase_cost = (df_mpc["grid_power"].clip(lower=0) * 0.001 * (time_step_min / 60) * df_mpc["unit_load_cost"]).sum()
        total_sale_revenue = (df_mpc["grid_power"].clip(upper=0).abs() * 0.001 * (time_step_min / 60) * df_mpc["unit_prod_price"]).sum()
        net_cost = total_purchase_cost - total_sale_revenue
        
        summary_lines = [
            "",
            "=" * 60,
            "                SIMULATION DAILY SUMMARY",
            "=" * 60,
            f" Total PV Production:           {total_pv_production_kwh:.2f} kWh",
            f" Total Grid Import:              {total_grid_import_kwh:.2f} kWh",
            f" Total Grid Export:              {total_grid_export_kwh:.2f} kWh",
            f" Total Consumption:             {total_consumption_kwh:.2f} kWh",
            "-" * 60,
            f" Base House Load:                {total_house_load_kwh:.2f} kWh",
            " Appliance Breakdown:"
        ]
        for name, energy in appliance_energy_dict.items():
            summary_lines.append(f"   - {name:<25} {energy:.2f} kWh")
        
        summary_lines.extend([
            "-" * 60,
            f" Final Battery SoC:              {df_mpc['batt_soc'].iloc[-1]*100:.1f} %",
            f" Self-Sufficiency (SSI):         {self_sufficiency*100:.1f} %",
            f" Self-Consumption (SCI):         {self_consumption*100:.1f} %",
            "-" * 60,
            f" Total Import Cost:          €   {total_purchase_cost:.2f}",
            f" Total Export Revenue:       €   {total_sale_revenue:.2f}",
            f" Net Cost (Cost - Rev):      €   {net_cost:.2f}",
            "=" * 60,
            ""
        ])

        summary_text = "\n".join(summary_lines)
        print(f"\n{summary_text}")

        with open(out_summary, "w") as f:
            f.write(summary_text + "\n")

    except Exception as e:
        print(f"   Error in MPC execution: {e}")

    print(f"\n[OK] Scenario {combo_name} completed.")

def main():
    parser = argparse.ArgumentParser(description="EMHASS Simulation Runner")
    parser.add_argument("--config", type=str, default="dynamic_miner_config.json", help="Name of the config file (inside configs/)")
    parser.add_argument("--dataset", type=str, default="scenario_2026_05_09.csv", help="Name of the dataset file (inside datasets/)")
    parser.add_argument("--run-all", action="store_true", help="Run all combinations of configs and datasets in batch mode")
    args = parser.parse_args()

    base_dir = current_dir

    if args.run_all:
        configs_dir = base_dir / "configs"
        datasets_dir = base_dir / "datasets"
        
        # Scan for all configuration and dataset files
        configs = [f.name for f in configs_dir.glob("*.json")]
        datasets = [f.name for f in datasets_dir.glob("*.csv")]
        
        print(f"--- BATCH MODE ---")
        print(f"Found {len(configs)} configs and {len(datasets)} datasets.")
        
        # Nested loop for Cartesian product of scenarios
        for cfg in configs:
            for ds in datasets:
                print(f"\n{'='*80}")
                print(f"Executing Batch Scenario: Config='{cfg}', Dataset='{ds}'")
                print(f"{'='*80}")
                try:
                    run_single_scenario(cfg, ds, base_dir)
                except Exception as e:
                    # Robust error handling for batch processing
                    print(f"[ERROR] Batch scenario {cfg} + {ds} failed: {e}", file=sys.stderr)
    else:
        # Single scenario execution
        run_single_scenario(args.config, args.dataset, base_dir)

if __name__ == "__main__":
    main()
