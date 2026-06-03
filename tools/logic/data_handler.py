import pandas as pd
import numpy as np
import pathlib

def generate_mock_base_data(start_date, time_step_min=30, days=2):
    """Generate a mock dataset with a sine wave PV curve and base load."""
    freq = f"{time_step_min}min"
    index = pd.date_range(start=start_date, periods=days * 24 * (60 // time_step_min), freq=freq, tz="UTC")
    
    # Simple sine wave for PV
    hour = index.hour + index.minute / 60
    pv = 5000 * np.maximum(0, np.sin((hour - 6) * np.pi / 12))
    
    # Base load: constant 500W + some peaks
    load = 500 + 1000 * (np.random.rand(len(index)) > 0.9).astype(float)
    
    # Prices
    load_cost = 0.25 * np.ones(len(index))
    prod_price = 0.10 * np.ones(len(index))
    
    return pd.DataFrame({"pv": pv, "load": load, "load_cost": load_cost, "prod_price": prod_price}, index=index)

def load_and_prepare_dataset(csv_path: str, time_step_min: int = 30) -> pd.DataFrame:
    """
    Load the dataset or generate it if it doesn't exist. 
    If forecast and actual columns are missing, it generates them by adding 
    realistic noise (sudden cloud drops or appliance spikes).
    """
    dataset_file = pathlib.Path(csv_path)
    
    if not dataset_file.exists():
        df = generate_mock_base_data(pd.Timestamp.now(tz="UTC").floor("D"), time_step_min=time_step_min, days=3)
    else:
        df = pd.read_csv(dataset_file, index_col=0, parse_dates=True)
        
    df.index = pd.to_datetime(df.index, utc=True)
    
    # Truncate to exactly 2 days (48 hours) for a 1-day simulation with a 1-day horizon
    expected_length = 2 * 24 * (60 // time_step_min)
    if len(df) > expected_length:
        df = df.iloc[:expected_length]
        
    # Add forecast/actual columns if not already split
    if "pv_forecast" not in df.columns:
        # Set a fixed seed to make noise reproducible
        np.random.seed(42)
        
        if "pv" in df.columns and "load" in df.columns:
            pv_forecast = df["pv"].copy()
            load_forecast = df["load"].copy()
        else:
            pv_forecast = df.iloc[:, 0].copy()
            load_forecast = df.iloc[:, 1].copy()
            
        # 1. PV Actual: 15% probability of significant drop due to clouds + small Gaussian noise
        cloud_factor = np.ones(len(df))
        cloud_indices = np.random.choice(len(df), size=int(len(df) * 0.15), replace=False)
        cloud_factor[cloud_indices] = np.random.uniform(0.3, 0.7, size=len(cloud_indices))
        
        pv_noise = np.random.normal(0, 120, size=len(df))
        pv_actual = pv_forecast * cloud_factor + pv_noise
        # No PV production at night
        pv_actual = np.where(pv_forecast > 10.0, np.maximum(0.0, pv_actual), 0.0)
        
        # 2. Load Actual: 10% probability of an appliance activation (spike up to 1.5kW) + background noise
        spike_load = np.zeros(len(df))
        spike_indices = np.random.choice(len(df), size=int(len(df) * 0.10), replace=False)
        spike_load[spike_indices] = np.random.uniform(500, 1500, size=len(spike_indices))
        
        load_noise = np.random.normal(0, 90, size=len(df))
        load_actual = np.maximum(0.0, load_forecast + load_noise + spike_load)
        
        df["pv_forecast"] = pv_forecast
        df["pv_actual"] = pv_actual
        df["load_forecast"] = load_forecast
        df["load_actual"] = load_actual
        
        # Remove old merged columns
        for col in ["pv", "load"]:
            if col in df.columns:
                df = df.drop(columns=[col])
                
        # Save so that noisy data remains consistent in future runs
        df.to_csv(dataset_file)
        
    return df
