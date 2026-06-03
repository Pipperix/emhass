import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pathlib

def generate_mpc_plot(results_csv_path: str, output_html_path: str):
    """
    Read the results CSV file and plot it using native Plotly. 
    The generated HTML is a single-window plot with a secondary SOC axis.
    """
    if not pathlib.Path(results_csv_path).exists():
        print(f"Error: Results file {results_csv_path} does not exist.")
        return

    df = pd.read_csv(results_csv_path, index_col=0, parse_dates=True)
    
    # Create a plot with a secondary y-axis for SOC
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Real PV (Orange)
    fig.add_trace(go.Scatter(x=df.index, y=df['pv_actual'], name="PV Actual (W)", 
                             line=dict(color='#ff9f43', width=2.5), fill='tozeroy', fillcolor='rgba(255, 159, 67, 0.1)'), secondary_y=False)
                             
    # Real House Load (Dashed Blue)
    fig.add_trace(go.Scatter(x=df.index, y=df['load_actual'], name="House Load Actual (W)", 
                             line=dict(color='#2e86de', width=2, dash='dash')), secondary_y=False)
                             
    # Assigned Miner Power (Teal Bar)
    fig.add_trace(go.Bar(x=df.index, y=df['miner_power'], name="Miner Power (W)", 
                         marker_color='#1dd1a1', opacity=0.85), secondary_y=False)
                         
    # Combined Grid Import/Export
    fig.add_trace(go.Scatter(x=df.index, y=df['grid_power'], name="Grid Power (W) [>0 Import]", 
                             line=dict(color='#ee5253', width=1.5)), secondary_y=False)
                             
    # Battery Power (Cyan)
    fig.add_trace(go.Scatter(x=df.index, y=df['batt_power'], name="Battery Power (W) [>0 Disch]", 
                             line=dict(color='#0abde3', width=2)), secondary_y=False)

    # Battery State of Charge (Dotted Grey on secondary axis)
    fig.add_trace(go.Scatter(x=df.index, y=df['batt_soc'] * 100, name="Battery SOC (%)", 
                             line=dict(color='#57606f', width=2, dash='dot')), secondary_y=True)

    # Calculate power axis extent for zero alignment
    all_power = pd.concat([
        df['pv_actual'], df['load_actual'], df['miner_power'],
        df['grid_power'], df['batt_power']
    ])
    min_power, max_power = all_power.min(), all_power.max()

    # Compute SOC axis lower bound so zeros coincide on both axes
    if min_power < 0:
        f = -min_power / (max_power - min_power)
        soc_min = -100.0 * f / (1.0 - f)
    else:
        soc_min = 0.0

    # General layout
    fig.update_layout(
        title={'text': "<b>EMHASS Standalone MPC Benchmark: Real-world Simulation</b>", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title="Time",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(255, 255, 255, 0.7)'),
        hovermode="x unified",
        margin=dict(t=120, b=50, l=60, r=60)
    )

    fig.update_yaxes(title_text="Power (Watts)", secondary_y=False)
    fig.update_yaxes(title_text="State of Charge (%)", range=[soc_min, 100], secondary_y=True)

    pathlib.Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html_path)
    print(f"Visualization created and saved at: {output_html_path}")


def generate_dayahead_plot(results_csv_path: str, output_html_path: str):
    """
    Read the Day-Ahead results CSV (EMHASS-native column names) and plot it.
    The generated HTML is a single-window plot with a secondary SOC axis.
    """
    if not pathlib.Path(results_csv_path).exists():
        print(f"Error: Results file {results_csv_path} does not exist.")
        return

    df = pd.read_csv(results_csv_path, index_col=0, parse_dates=True)

    # Map EMHASS-native columns to display names
    rename_map = {
        'P_PV': 'pv_actual',
        'P_Load': 'load_actual',
        'P_deferrable0': 'miner_power',
        'P_grid': 'grid_power',
        'P_batt': 'batt_power',
        'SOC_opt': 'batt_soc',
    }
    df.rename(columns=rename_map, inplace=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # PV (Orange)
    fig.add_trace(go.Scatter(x=df.index, y=df['pv_actual'], name="PV Actual (W)",
                             line=dict(color='#ff9f43', width=2.5), fill='tozeroy', fillcolor='rgba(255, 159, 67, 0.1)'), secondary_y=False)

    # House Load (Dashed Blue)
    fig.add_trace(go.Scatter(x=df.index, y=df['load_actual'], name="House Load Actual (W)",
                             line=dict(color='#2e86de', width=2, dash='dash')), secondary_y=False)

    # Miner Power (Teal Bar)
    fig.add_trace(go.Bar(x=df.index, y=df['miner_power'], name="Miner Power (W)",
                         marker_color='#1dd1a1', opacity=0.85), secondary_y=False)

    # Grid Power
    fig.add_trace(go.Scatter(x=df.index, y=df['grid_power'], name="Grid Power (W) [>0 Import]",
                             line=dict(color='#ee5253', width=1.5)), secondary_y=False)

    # Battery Power (Cyan)
    fig.add_trace(go.Scatter(x=df.index, y=df['batt_power'], name="Battery Power (W) [>0 Disch]",
                             line=dict(color='#0abde3', width=2)), secondary_y=False)

    # Battery SOC (secondary axis)
    fig.add_trace(go.Scatter(x=df.index, y=df['batt_soc'] * 100, name="Battery SOC (%)",
                             line=dict(color='#57606f', width=2, dash='dot')), secondary_y=True)

    # Zero-alignment calculation
    all_power = pd.concat([
        df['pv_actual'], df['load_actual'], df['miner_power'],
        df['grid_power'], df['batt_power']
    ])
    min_power, max_power = all_power.min(), all_power.max()

    if min_power < 0:
        f = -min_power / (max_power - min_power)
        soc_min = -100.0 * f / (1.0 - f)
    else:
        soc_min = 0.0

    fig.update_layout(
        title={'text': "<b>EMHASS Day-Ahead Open-Loop Optimization</b>", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title="Time",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(255, 255, 255, 0.7)'),
        hovermode="x unified",
        margin=dict(t=120, b=50, l=60, r=60)
    )

    fig.update_yaxes(title_text="Power (Watts)", secondary_y=False)
    fig.update_yaxes(title_text="State of Charge (%)", range=[soc_min, 100], secondary_y=True)

    pathlib.Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html_path)
    print(f"Day-Ahead visualization created and saved at: {output_html_path}")
