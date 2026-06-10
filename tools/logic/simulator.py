"""
EMHASS Simulation Module.
Two simulation modes:
  - Day-Ahead: open-loop planning over the entire horizon (based on forecasts).
  - MPC: closed-loop timestep-by-timestep control (decisions based on forecasts, 
    but battery state update based on real data).
"""

import asyncio
import logging
import pathlib
import warnings

import numpy as np
import pandas as pd

from emhass.optimization import Optimization
from emhass.utils import (
    build_config,
    build_params,
    get_logger,
    get_root,
    get_yaml_parse,
)

# Disable CVXPY/Pandas deprecation warnings to keep output clean
warnings.filterwarnings("ignore")


class EmhassSimulator:
    """
    Wrapper that replicates EMHASS initialization and exposes 
    Day-Ahead and MPC simulation methods.
    """

    def __init__(self, config: dict):
        self.config = config
        self.emhass_root = get_root(__file__, num_parent=3)

        # Setup paths identical to the benchmark (pathlib.Path, not strings)
        self.emhass_conf = {
            "data_path": self.emhass_root / "data/",
            "root_path": self.emhass_root / "src/emhass/",
            "config_path": self.emhass_root / "config.json",
            "defaults_path": self.emhass_root / "src/emhass/data/config_defaults.json",
            "associations_path": self.emhass_root / "src/emhass/data/associations.csv",
        }

        # Create logger
        self.logger, _ = get_logger("emhass_simulator", self.emhass_conf, save_to_file=False)
        self.logger.setLevel(logging.WARNING)

        # Asynchronous initialization of the EMHASS pipeline
        self.opt = asyncio.get_event_loop().run_until_complete(self._init_optimization())

    async def _init_optimization(self) -> Optimization:
        """
        Initialize the EMHASS Optimization object using the native pipeline:
        build_config → param overrides → build_params → get_yaml_parse → Optimization.
        """
        # 1. Load defaults from config_defaults.json
        config = await build_config(
            self.emhass_conf,
            self.logger,
            self.emhass_conf["defaults_path"],
        )

        # 2. Map dynamic Deferrable Loads from Sim Config
        def_loads = self.config["deferrable_loads"]
        num_def = len(def_loads)
        
        config["number_of_deferrable_loads"] = num_def
        config["nominal_power_of_deferrable_loads"] = [d["nominal_power"] for d in def_loads]
        config["minimum_power_of_deferrable_loads"] = [d["minimum_power"] for d in def_loads]
        config["treat_deferrable_load_as_semi_cont"] = [d["treat_as_semi_cont"] for d in def_loads]
        config["set_deferrable_load_single_constant"] = [d["set_single_constant"] for d in def_loads]
        config["operating_hours_of_each_deferrable_load"] = [d["operating_hours"] for d in def_loads]

        # 3. Override System parameters (Fail-fast: no .get() with defaults)
        config["set_use_battery"] = self.config["set_use_battery"]
        config["battery_nominal_energy_capacity"] = self.config["battery_nominal_energy_capacity"]
        config["battery_minimum_state_of_charge"] = self.config["battery_minimum_state_of_charge"]
        config["battery_maximum_state_of_charge"] = self.config["battery_maximum_state_of_charge"]
        config["battery_target_state_of_charge"] = self.config["battery_target_state_of_charge"]
        config["battery_discharge_efficiency"] = self.config["battery_discharge_efficiency"]
        config["battery_charge_efficiency"] = self.config["battery_charge_efficiency"]
        config["battery_discharge_power_max"] = self.config["battery_discharge_power_max"]
        config["battery_charge_power_max"] = self.config["battery_charge_power_max"]
        
        config["battery_soc_deficit_threshold"] = self.config["battery_soc_deficit_threshold"]
        config["battery_soc_deficit_cost"] = self.config["battery_soc_deficit_cost"]
        config["weight_battery_discharge"] = self.config["weight_battery_discharge"]
        config["weight_battery_charge"] = self.config["weight_battery_charge"]

        config["optimization_time_step"] = self.config["time_step_min"]
        config["costfun"] = self.config["costfun"]
        config["set_total_pv_sell"] = self.config["set_total_pv_sell"]
        config["set_nocharge_from_grid"] = self.config["set_nocharge_from_grid"]

        # 4. Build params and extract configurations
        params = await build_params(self.emhass_conf, {}, config, self.logger)
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(params, self.logger)

        # 5. Initialize the Optimization object
        window_steps = self.config["window_size_hours"] * (60 // self.config["time_step_min"])
        
        opt = Optimization(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "unit_load_cost",
            "unit_prod_price",
            self.config["costfun"],
            self.emhass_conf,
            self.logger,
            num_timesteps=window_steps,
        )

        return opt

    def simulate_day_ahead(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """Simulate Day-Ahead open-loop execution."""
        df_input = df_data.copy()

        df_input_data = pd.DataFrame(
            {
                "p_pv_forecast": df_input["pv_forecast"].values,
                "P_load_forecast": df_input["load_forecast"].values,
                "unit_load_cost": df_input["load_cost"].values,
                "unit_prod_price": df_input["prod_price"].values,
            },
            index=df_input.index,
        )

        # Dynamic virtual costs for each load
        self.opt.optim_conf["cost_forecast_per_deferrable_load"] = [
            [d["virtual_cost"]] * len(df_input_data)
            for d in self.config["deferrable_loads"]
        ]

        df_opt = self.opt.perform_optimization(
            df_input_data,
            df_input["pv_forecast"].values,
            df_input["load_forecast"].values,
            df_input["load_cost"].values,
            df_input["prod_price"].values,
            soc_init=self.config["initial_soc"],
            soc_final=self.config["battery_target_state_of_charge"],
            def_total_hours=[d["operating_hours"] for d in self.config["deferrable_loads"]],
        )

        return df_opt

    def simulate_mpc(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """Simulate the dynamic behavior of Model Predictive Control (MPC)."""
        mpc_results = []
        time_step = self.config["time_step_min"]
        window_size = self.config["window_size_hours"] * (60 // time_step)
        sim_length = len(df_data) - window_size

        if sim_length <= 0:
            raise ValueError(f"Dataset too short for configured window_size.")

        # Initial state
        current_soc = self.config["initial_soc"]
        soc_final = self.config["battery_target_state_of_charge"]
        battery_capacity_wh = self.config["battery_nominal_energy_capacity"]
        dt_hours = time_step / 60.0

        # Battery limits
        soc_min = self.config["battery_minimum_state_of_charge"]
        soc_max = self.config["battery_maximum_state_of_charge"]
        eff_disch = self.config["battery_discharge_efficiency"]
        eff_ch = self.config["battery_charge_efficiency"]
        max_p_disch_inverter = self.config["battery_discharge_power_max"]
        max_p_ch_inverter = self.config["battery_charge_power_max"]

        for t in range(sim_length):
            now = df_data.index[t]
            window = df_data.iloc[t : t + window_size]

            df_input_data = pd.DataFrame(
                {
                    "p_pv_forecast": window["pv_forecast"].values,
                    "P_load_forecast": window["load_forecast"].values,
                    "unit_load_cost": window["load_cost"].values,
                    "unit_prod_price": window["prod_price"].values,
                },
                index=window.index,
            )

            # Dynamic virtual costs for each load
            self.opt.optim_conf["cost_forecast_per_deferrable_load"] = [
                [d["virtual_cost"]] * len(window)
                for d in self.config["deferrable_loads"]
            ]

            try:
                p_pv_mpc = df_input_data["p_pv_forecast"].copy()
                p_pv_mpc.iloc[0] = window["pv_actual"].iloc[0]
                
                p_load_mpc = df_input_data["P_load_forecast"].copy()
                p_load_mpc.iloc[0] = window["load_actual"].iloc[0]

                opt_res = self.opt.perform_naive_mpc_optim(
                    df_input_data,
                    p_pv=p_pv_mpc,
                    p_load=p_load_mpc,
                    prediction_horizon=len(window),
                    soc_init=current_soc,
                    soc_final=soc_final,
                    def_total_hours=[d["operating_hours"] for d in self.config["deferrable_loads"]],
                )
            except Exception as e:
                self.logger.error(f"Optimization failed at {now}: {e}")
                break

            if opt_res is None or opt_res.empty:
                break

            # Extract decisions for all deferrable loads
            current_res = opt_res.iloc[0]
            opt_grid = current_res["P_grid"]
            
            # Extract and store power for each deferrable load
            def_powers = {}
            total_def_power = 0.0
            for i, load in enumerate(self.config["deferrable_loads"]):
                col_name = f"P_deferrable{i}"
                power = current_res[col_name]
                
                # TODO: Implement workaround for purely opportunistic static loads
                if load.get("force_quantization", False):
                    if power >= (load["nominal_power"] * 0.95):
                        power = load["nominal_power"]
                    else:
                        power = 0.0
                        
                def_powers[load["name"]] = power
                total_def_power += power

            # === REAL WORLD SIMULATION ===
            actual_pv = window["pv_actual"].iloc[0]
            actual_load = window["load_actual"].iloc[0]

            # Nodal balance: P_batt = P_load + sum(P_def) - P_pv - P_grid
            real_p_batt = actual_load + total_def_power - actual_pv - opt_grid

            # Physics-based energy limits
            max_disch_energy_wh = max(0.0, (current_soc - soc_min) * battery_capacity_wh)
            max_ch_energy_wh = max(0.0, (soc_max - current_soc) * battery_capacity_wh)

            max_p_disch_soc = (max_disch_energy_wh * eff_disch) / dt_hours
            max_p_ch_soc = max_ch_energy_wh / (dt_hours * eff_ch)

            p_batt_max_disch = min(max_p_disch_inverter, max_p_disch_soc)
            p_batt_max_ch = min(max_p_ch_inverter, max_p_ch_soc)

            clipped_p_batt = min(p_batt_max_disch, max(-p_batt_max_ch, real_p_batt))
            actual_grid = opt_grid + (real_p_batt - clipped_p_batt)

            # thermodynamic SOC update
            if clipped_p_batt > 0:  energy_change_wh = clipped_p_batt * dt_hours / eff_disch
            else: energy_change_wh = clipped_p_batt * dt_hours * eff_ch

            new_soc = current_soc - (energy_change_wh / battery_capacity_wh)
            new_soc = max(soc_min, min(soc_max, new_soc))

            # Save results
            step_result = {
                "timestamp": now,
                "pv_actual": actual_pv,
                "load_actual": actual_load,
                "grid_power": actual_grid,
                "batt_power": clipped_p_batt,
                "batt_soc": new_soc,
            }
            step_result.update(def_powers) # Include each individual load power
            mpc_results.append(step_result)

            current_soc = new_soc

        return pd.DataFrame(mpc_results).set_index("timestamp")
