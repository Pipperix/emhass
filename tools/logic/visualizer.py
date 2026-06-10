import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pathlib

def generate_mpc_plot(results_csv_path: str, output_html_path: str):
    """
    Read the results CSV file and plot it using native Plotly. 
    Supports dynamic number of deferrable loads using stacked bars.
    """
    if not pathlib.Path(results_csv_path).exists():
        print(f"Error: Results file {results_csv_path} does not exist.")
        return

    df = pd.read_csv(results_csv_path, index_col=0, parse_dates=True)
    
    # Identify deferrable load columns (those not in the standard list)
    standard_cols = ['pv_actual', 'load_actual', 'grid_power', 'batt_power', 'batt_soc']
    def_load_cols = [c for c in df.columns if c not in standard_cols]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Real PV (Orange)
    fig.add_trace(go.Scatter(x=df.index, y=df['pv_actual'], name="PV Actual (W)", 
                             line=dict(color='#ff9f43', width=2.5), fill='tozeroy', fillcolor='rgba(255, 159, 67, 0.1)'), secondary_y=False)
                             
    # 2. Real House Load (Dashed Blue)
    fig.add_trace(go.Scatter(x=df.index, y=df['load_actual'], name="House Load Actual (W)", 
                             line=dict(color='#2e86de', width=2, dash='dash')), secondary_y=False)
                             
    # 3. Dynamic Deferrable Loads (Overlaid Bars)
    colors = ['#1dd1a1', '#2e86de', '#ff9f43', '#9b59b6', '#ee5253', '#0abde3', '#10ac84', '#5f27cd']
    for i, col in enumerate(def_load_cols):
        color = colors[i % len(colors)]
        fig.add_trace(go.Bar(x=df.index, y=df[col], name=f"{col} (W)", 
                             marker_color=color, opacity=0.35), secondary_y=False)
                         
    # 4. Grid Power
    fig.add_trace(go.Scatter(x=df.index, y=df['grid_power'], name="Grid Power (W) [>0 Import]", 
                             line=dict(color='#ee5253', width=1.5)), secondary_y=False)
                             
    # 5. Battery Power
    fig.add_trace(go.Scatter(x=df.index, y=df['batt_power'], name="Battery Power (W) [>0 Disch]", 
                             line=dict(color='#0abde3', width=2)), secondary_y=False)

    # 6. Battery SOC
    fig.add_trace(go.Scatter(x=df.index, y=df['batt_soc'] * 100, name="Battery SOC (%)", 
                             line=dict(color='#57606f', width=2, dash='dot')), secondary_y=True)

    # Coordinate y-axes ranges to align their zero lines perfectly
    all_power_cols = [c for c in df.columns if c != 'batt_soc']
    min_power = df[all_power_cols].min().min()
    max_power = df[all_power_cols].max().max()

    y1_max = max_power * 1.1 if max_power > 0 else 1000.0
    y1_min = min_power * 1.1 if min_power < 0 else -y1_max * 0.05
    y2_max = 105.0
    y2_min = y2_max * (y1_min / y1_max)

    fig.update_layout(
        title={'text': "<b>EMHASS Standalone MPC Benchmark: Multi-Appliance Simulation</b>", 'x':0.5, 'xanchor': 'center', 'y': 0.98},
        xaxis_title="Time",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, bgcolor='rgba(255, 255, 255, 0.7)'),
        hovermode="x unified",
        barmode='overlay',
        margin=dict(t=160, b=80, l=60, r=60),
        height=700
    )

    fig.update_yaxes(title_text="Power (Watts)", range=[y1_min, y1_max], secondary_y=False)
    fig.update_yaxes(title_text="State of Charge (%)", range=[y2_min, y2_max], secondary_y=True)



    pathlib.Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html_path)
    print(f"MPC Visualization created: {output_html_path}")


def generate_dayahead_plot(results_csv_path: str, output_html_path: str, config: dict):
    """
    Read the Day-Ahead results CSV (EMHASS-native column names) and plot it.
    Mapping deferrable loads from config to match EMHASS indexed column names.
    """
    if not pathlib.Path(results_csv_path).exists():
        print(f"Error: Results file {results_csv_path} does not exist.")
        return

    df = pd.read_csv(results_csv_path, index_col=0, parse_dates=True)

    # Map EMHASS-native standard columns
    rename_map = {
        'P_PV': 'pv_actual',
        'P_Load': 'load_actual',
        'P_grid': 'grid_power',
        'P_batt': 'batt_power',
        'SOC_opt': 'batt_soc',
    }
    
    # Map indexed deferrable loads to their names from config
    for i, load in enumerate(config["deferrable_loads"]):
        rename_map[f'P_deferrable{i}'] = load["name"]

    df.rename(columns=rename_map, inplace=True)
    
    # Filter the DataFrame to keep only the mapped columns (removing solver diagnostic variables)
    cols_to_keep = [c for c in rename_map.values() if c in df.columns]
    df = df[cols_to_keep]
    
    # Re-use the same logic as MPC for plotting
    generate_mpc_plot_from_df(df, output_html_path, "EMHASS Day-Ahead Multi-Appliance Optimization")

def generate_mpc_plot_from_df(df, output_html_path, title):
    """Helper to generate plot from a prepared DataFrame."""
    standard_cols = ['pv_actual', 'load_actual', 'grid_power', 'batt_power', 'batt_soc']
    def_load_cols = [c for c in df.columns if c not in standard_cols and not c.startswith('cost_')]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df['pv_actual'], name="PV Actual (W)", line=dict(color='#ff9f43', width=2.5), fill='tozeroy', fillcolor='rgba(255, 159, 67, 0.1)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df['load_actual'], name="House Load Actual (W)", line=dict(color='#2e86de', width=2, dash='dash')), secondary_y=False)
    
    colors = ['#1dd1a1', '#2e86de', '#ff9f43', '#9b59b6', '#ee5253', '#0abde3', '#10ac84', '#5f27cd']
    for i, col in enumerate(def_load_cols):
        fig.add_trace(go.Bar(x=df.index, y=df[col], name=f"{col} (W)", marker_color=colors[i % len(colors)], opacity=0.35), secondary_y=False)

    fig.add_trace(go.Scatter(x=df.index, y=df['grid_power'], name="Grid Power (W) [>0 Import]", line=dict(color='#ee5253', width=1.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df['batt_power'], name="Battery Power (W) [>0 Disch]", line=dict(color='#0abde3', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df['batt_soc'] * 100, name="Battery SOC (%)", line=dict(color='#57606f', width=2, dash='dot')), secondary_y=True)

    # Coordinate y-axes ranges to align their zero lines perfectly
    all_power_cols = [c for c in df.columns if c != 'batt_soc' and not c.startswith('cost_')]
    min_power = df[all_power_cols].min().min()
    max_power = df[all_power_cols].max().max()

    y1_max = max_power * 1.1 if max_power > 0 else 1000.0
    y1_min = min_power * 1.1 if min_power < 0 else -y1_max * 0.05
    y2_max = 105.0
    y2_min = y2_max * (y1_min / y1_max)

    fig.update_layout(title={'text': f"<b>{title}</b>", 'x':0.5, 'xanchor': 'center', 'y': 0.98}, 
                      template="plotly_white", barmode='overlay', hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(255, 255, 255, 0.7)'),
                      margin=dict(t=160, b=80, l=60, r=60),
                      height=700)
    fig.update_yaxes(title_text="Power (Watts)", range=[y1_min, y1_max], secondary_y=False)
    fig.update_yaxes(title_text="State of Charge (%)", range=[y2_min, y2_max], secondary_y=True)


    pathlib.Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html_path)
    print(f"Visualization created: {output_html_path}")

