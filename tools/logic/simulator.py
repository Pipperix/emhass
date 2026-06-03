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

        This ensures all parameters are validated, partitioned, 
        and converted exactly as in a production environment.
        """
        # 1. Load defaults from config_defaults.json (similar to web_server/command_line)
        config = await build_config(
            self.emhass_conf,
            self.logger,
            self.emhass_conf["defaults_path"],
        )

        # 2. Override specific simulation parameters
        # TODO: Use config to automatically override all parameters
        config["number_of_deferrable_loads"] = 1
        config["nominal_power_of_deferrable_loads"] = [
            self.config.get("miner_max_power", 6000.0)
        ]
        config["minimum_power_of_deferrable_loads"] = [0.0]
        config["treat_deferrable_load_as_semi_cont"] = self.config.get(
            "treat_deferrable_load_as_semi_cont", [False] # If True, the load can only be ON at full power or OFF, otherwise it can be partially loaded
        )
        config["set_deferrable_load_single_constant"] = self.config.get(
            "set_deferrable_load_single_constant", [False] # If True, the load must be ON for a single contiguous block of time, otherwise it can be split in multiple blocks
        )
        config["operating_hours_of_each_deferrable_load"] = self.config.get(
            "deferrable_load_operating_hours", [0.0] # Minimum running hours for the deferrable load
        )

        # Battery
        config["set_use_battery"] = True
        config["battery_nominal_energy_capacity"] = self.config.get(
            "battery_nominal_energy_capacity", 5000.0
        )
        config["battery_minimum_state_of_charge"] = self.config.get(
            "battery_minimum_state_of_charge", 0.3
        )
        config["battery_maximum_state_of_charge"] = self.config.get(
            "battery_maximum_state_of_charge", 0.9
        )
        config["battery_target_state_of_charge"] = self.config.get(
            "battery_target_state_of_charge", 0.6
        )
        config["battery_discharge_efficiency"] = self.config.get(
            "battery_discharge_efficiency", 0.95
        )
        config["battery_charge_efficiency"] = self.config.get(
            "battery_charge_efficiency", 0.95
        )
        config["battery_discharge_power_max"] = self.config.get(
            "battery_discharge_power_max", 2000.0
        )
        config["battery_charge_power_max"] = self.config.get(
            "battery_charge_power_max", 2000.0
        )

        # SOC penalty and battery cycle cost
        # battery_soc_deficit_cost: penalty (€/kWh) when SOC falls below threshold
        # weight_battery_discharge/charge: wear cost per discharge/charge cycle
        config["battery_soc_deficit_threshold"] = self.config.get(
            "battery_soc_deficit_threshold", 0.40
        )
        config["battery_soc_deficit_cost"] = self.config.get(
            "battery_soc_deficit_cost", 0.0
        )
        config["weight_battery_discharge"] = self.config.get(
            "weight_battery_discharge", 0.0
        )
        config["weight_battery_charge"] = self.config.get(
            "weight_battery_charge", 0.0
        )

        config["optimization_time_step"] = self.config.get("time_step_min", 30) # minutes per optimization step
        config["costfun"] = self.config.get("costfun", "profit")
        config["set_total_pv_sell"] = self.config.get("set_total_pv_sell", False) # When true: all excess PV is sold to the grid [Gross-metering]; when false: excess PV can be sold but is not forced to be [Net-metering] 
        config["set_nocharge_from_grid"] = self.config.get("set_nocharge_from_grid", False)

        # 3. Build params and extract configurations
        params = await build_params(self.emhass_conf, {}, config, self.logger)
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(params, self.logger)

        # 4. Initialize the Optimization object once outside the loop
        # Optimization is stateless, data is passed as input to the perform_naive_mpc_optim method at each step
        window_steps = self.config.get("window_size_hours", 24) * (
            60 // self.config.get("time_step_min", 30)
        )
        opt = Optimization(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "unit_load_cost",
            "unit_prod_price",
            self.config.get("costfun", "profit"),
            self.emhass_conf,
            self.logger,
            num_timesteps=window_steps, # TODO: Learn more about this
        )

        return opt

    def simulate_day_ahead(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate Day-Ahead open-loop execution.

        Calculates the ideal plan once for the entire horizon,
        based exclusively on forecast data. In reality, this
        optimization is run once per day.
        """
        df_input = df_data.copy()

        # Prepare DataFrame in the format expected by the benchmark:
        # columns renamed to names used as var_load_cost / var_prod_price
        df_input_data = pd.DataFrame(
            {
                "p_pv_forecast": df_input["pv_forecast"].values,
                "P_load_forecast": df_input["load_forecast"].values,
                "unit_load_cost": df_input["load_cost"].values,
                "unit_prod_price": df_input["prod_price"].values,
            },
            index=df_input.index,
        )

        # Set the miner's virtual cost (as in benchmark line 177)
        self.opt.optim_conf["cost_forecast_per_deferrable_load"] = [
            [c] * len(df_input_data)
            for c in self.config.get("deferrable_load_virtual_cost", [0.01])
        ]

        # Execute Day-Ahead optimization
        # Note: perform_dayahead_forecast_optim does not accept soc_init/soc_final/def_total_hours,
        # so we use perform_optimization directly with all necessary parameters
        df_opt = self.opt.perform_optimization(
            df_input_data,
            df_input["pv_forecast"].values,
            df_input["load_forecast"].values,
            df_input["load_cost"].values,
            df_input["prod_price"].values,
            soc_init=self.config.get("initial_soc", 0.4),
            soc_final=self.config.get("battery_target_state_of_charge", 0.6),
            def_total_hours=self.config.get("deferrable_load_operating_hours", [0.0]),
        )

        return df_opt

    def simulate_mpc(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate the dynamic behavior of Model Predictive Control (MPC).

        At each timestep:
          1. The optimizer receives FORECAST data and calculates the optimal plan.
          2. Decisions are extracted only for the first timestep.
          3. Decisions are applied to the REAL world ('actual' data).
          4. Battery SOC is updated with real physics.
          5. The new SOC is passed to the optimizer in the next step.

        This is the core of the simulation: it shows how discrepancies between
        forecasts and reality impact system performance.
        """
        mpc_results = []
        time_step = self.config.get("time_step_min", 30)
        window_size = self.config.get("window_size_hours", 24) * (60 // time_step)
        sim_length = len(df_data) - window_size

        if sim_length <= 0:
            raise ValueError(
                f"Dataset too short ({len(df_data)} timesteps) for the "
                f"configured window_size ({window_size} timesteps)."
            )

        # Initial simulation state
        current_soc = self.config.get("initial_soc", 0.4)
        soc_final = self.config.get("battery_target_state_of_charge", 0.6)
        battery_capacity_wh = self.config.get("battery_nominal_energy_capacity", 5000.0)
        dt_hours = time_step / 60.0

        # Battery physical limits
        soc_min = self.config.get("battery_minimum_state_of_charge", 0.3)
        soc_max = self.config.get("battery_maximum_state_of_charge", 0.9)
        eff_disch = self.config.get("battery_discharge_efficiency", 0.95)
        eff_ch = self.config.get("battery_charge_efficiency", 0.95)
        max_p_disch_inverter = self.config.get("battery_discharge_power_max", 2000.0)
        max_p_ch_inverter = self.config.get("battery_charge_power_max", 2000.0)

        for t in range(sim_length):
            now = df_data.index[t]
            window = df_data.iloc[t : t + window_size]

            # Prepare DataFrame in the format identical to the benchmark
            df_input_data = pd.DataFrame(
                {
                    "p_pv_forecast": window["pv_forecast"].values,
                    "P_load_forecast": window["load_forecast"].values,
                    "unit_load_cost": window["load_cost"].values,
                    "unit_prod_price": window["prod_price"].values,
                },
                index=window.index,
            )

            # Miner's virtual cost for this window (as in benchmark line 177)
            self.opt.optim_conf["cost_forecast_per_deferrable_load"] = [
                [c] * len(window)
                for c in self.config.get("deferrable_load_virtual_cost", [0.01])
            ]

            # Execute MPC using EMHASS native wrapper (as in benchmark line 181)
            try:
                # TODO: Review this code next
                # Inject actual values at t=0 for closed-loop feedback
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
                    def_total_hours=self.config.get(
                        "deferrable_load_operating_hours", [0.0]
                    ),
                )
            except Exception as e:
                self.logger.error(f"Optimization failed at timestep {t} ({now}): {e}")
                break

            if opt_res is None or opt_res.empty:
                self.logger.error(f"Empty results at timestep {t} ({now})")
                break

            # Extract algorithm decisions for the first timestep only
            current_res = opt_res.iloc[0]
            opt_miner_power = current_res["P_deferrable0"]
            opt_grid = current_res["P_grid"]

            # === REAL WORLD SIMULATION ===
            # Here we apply optimizer decisions to ACTUAL data
            actual_pv = window["pv_actual"].iloc[0]
            actual_load = window["load_actual"].iloc[0]

            # Nodal energy balance: P_batt = P_load + P_miner - P_pv - P_grid
            # (positive = battery discharging, negative = battery charging)
            real_p_batt = actual_load + opt_miner_power - actual_pv - opt_grid

            # Energy limits based on current battery chemical state.
            # This prevents the battery from providing "ghost energy".
            max_disch_energy_wh = max(0.0, (current_soc - soc_min) * battery_capacity_wh)
            max_ch_energy_wh = max(0.0, (soc_max - current_soc) * battery_capacity_wh)

            # Conversion to maximum power deliverable/storable in the timestep
            max_p_disch_soc = (max_disch_energy_wh * eff_disch) / dt_hours
            max_p_ch_soc = max_ch_energy_wh / (dt_hours * eff_ch)

            # Final limit: min(inverter capacity, available energy)
            p_batt_max_disch = min(max_p_disch_inverter, max_p_disch_soc)
            p_batt_max_ch = min(max_p_ch_inverter, max_p_ch_soc)

            # Clipping battery power to real physical limits
            clipped_p_batt = min(p_batt_max_disch, max(-p_batt_max_ch, real_p_batt))

            # Unbalance not covered by the battery is absorbed by the grid
            actual_grid = opt_grid + (real_p_batt - clipped_p_batt)

            # Thermodynamic SOC update
            if clipped_p_batt > 0:  # Discharging
                energy_change_wh = clipped_p_batt * dt_hours / eff_disch
            else:  # Charging
                energy_change_wh = clipped_p_batt * dt_hours * eff_ch

            new_soc = current_soc - (energy_change_wh / battery_capacity_wh)
            new_soc = max(soc_min, min(soc_max, new_soc))

            # Saving results for this timestep
            mpc_results.append(
                {
                    "timestamp": now,
                    "pv_actual": actual_pv,
                    "load_actual": actual_load,
                    "miner_power": opt_miner_power,
                    "grid_power": actual_grid,
                    "batt_power": clipped_p_batt,
                    "batt_soc": new_soc,
                }
            )

            # Update state for the next timestep
            current_soc = new_soc

            # Progress log every 10 steps
            if t % 10 == 0:
                self.logger.info(
                    f"Step {t}/{sim_length}: Miner={opt_miner_power:.1f}W, "
                    f"SOC={new_soc:.3f}, Grid={actual_grid:.1f}W"
                )

        return pd.DataFrame(mpc_results).set_index("timestamp")
