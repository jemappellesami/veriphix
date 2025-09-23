import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------- Helpers ----------

def load_table():
    path = Path("circuits/table.json")
    if not path.exists():
        raise FileNotFoundError("Missing file: circuits/table.json")
    with path.open() as f:
        return json.load(f)

def get_info(circuit_name: str, table: dict, protocol:str) -> tuple[int, float, float]:
    """
    Returns (correct_value_rounded, bqp_error, threshold),
    using your abs-rounding for the threshold formula.
    """
    raw = float(table[circuit_name])
    rounded = round(raw)
    abs_round = raw if raw <= 0.5 else 1 - raw
    denom = (2 - 2 * abs_round)
    k = 2 if protocol=='FK12' else 286
    threshold = 1/k * (1 - 2 * abs_round) / denom if denom != 0 else float("nan")
    return rounded, raw, threshold

def clip_0_100(x):
    try:
        return int(np.clip(int(x), 0, 100))
    except Exception:
        return 0

def filtered_subset(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """Subset to selected protocol and malicious noise model."""
    return df[
        (df["protocol"] == protocol) &
        (df["noise_model"] == "malicious")
    ].copy()

def build_figure(data: pd.DataFrame, protocol: str, circuit: str, param: float) -> go.Figure:
    df = data[
        (data["protocol"] == protocol) &
        (data["noise_model"] == "malicious") &
        (data["circuit_label"] == circuit) &
        (data["parameter"] == param)
    ]
    if df.empty:
        raise ValueError(f"No data for protocol '{protocol}', circuit '{circuit}' and parameter {param}")
    row = df.iloc[0]

    # Bar: "0" = 100 - computation_outcomes_count, "1" = computation_outcomes_count
    bar_y = [
        clip_0_100(100 - row["computation_outcomes_count"]),
        clip_0_100(row["computation_outcomes_count"]),
    ]
    # Pie: red = n_failed_test_rounds, green = 100 - n_failed_test_rounds
    pie_vals = [
        clip_0_100(row["n_failed_test_rounds"]),
        clip_0_100(100 - row["n_failed_test_rounds"]),
    ]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "domain"}]],
        subplot_titles=("Computation Outcomes", "Failed Test Rounds")
    )

    fig.add_trace(go.Bar(
        x=["0", "1"],
        y=bar_y,
        text=bar_y,
        textposition="outside",
        name="Outcomes",
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        labels=["Failed (red)", "Passed (green)"],
        values=pie_vals,
        hole=0.3,
        marker=dict(colors=["red", "green"]),
        sort=False,
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(height=520, margin=dict(l=40, r=40, t=60, b=40))
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_xaxes(title="Outcome", row=1, col=1)
    return fig


# ---------- Dash App ----------

def create_dash_app(df: pd.DataFrame, circuit_table: dict):
    app = dash.Dash(__name__)
    app.title = "Interactive Circuit Viewer"

    # Preprocess once
    df["parameter"] = pd.to_numeric(df["parameter"], errors="coerce")

    app.layout = html.Div([
        html.H2("Quantum Circuit Results Viewer"),

        html.Div([
            html.Label("Protocol:", style={"marginRight": "0.5em"}),
            dcc.RadioItems(
                id="protocol-radio",
                options=[{"label": "FK12", "value": "FK12"},
                         {"label": "Dummyless", "value": "Dummyless"}],
                value="FK12",
                inline=True,
            ),
        ], style={"marginBottom": "0.75em"}),

        html.Div([
            html.Button("🎲 Random Circuit", id="random-btn", n_clicks=0),
            html.Span("  "),
            html.Strong("Selected Circuit:"),
            html.Span(id="circuit-name", style={"marginLeft": "0.5em"})
        ], style={"marginBottom": "1em"}),

        html.Div([
            html.Label("Parameter:"),
            dcc.Dropdown(id="param-dropdown", style={"width": "200px"})
        ], style={"marginBottom": "1em"}),

        dcc.Graph(id="result-figure"),

        html.Div(id="info-panel", style={"marginTop": "1em", "fontSize": "16px"})
    ], style={"padding": "2em", "fontFamily": "sans-serif"})

    # Single coordinator callback:
    # - Keeps current circuit when protocol changes.
    # - Randomizes only when the random button triggers the callback.
    # - Preserves the current parameter if still available; otherwise picks first available.
    @app.callback(
        Output("circuit-name", "children"),
        Output("param-dropdown", "options"),
        Output("param-dropdown", "value"),
        Input("protocol-radio", "value"),
        Input("random-btn", "n_clicks"),
        State("circuit-name", "children"),
        State("param-dropdown", "value"),
        prevent_initial_call=False
    )
    def pick_circuit_and_params(protocol, n_clicks, current_circuit, current_param):
        sub = filtered_subset(df, protocol)
        circuits = sorted(sub["circuit_label"].dropna().unique())
        if not circuits:
            raise ValueError(f"No circuits found for protocol={protocol} and noise_model=malicious.")

        # Decide which circuit to use:
        # - If Random button triggered, pick a random one.
        # - Else, keep the current circuit if it exists in this protocol; if not, fall back to first (deterministic).
        trigger = ctx.triggered_id
        if trigger == "random-btn":
            selected = random.choice(circuits)
        else:
            selected = current_circuit if current_circuit in circuits else circuits[0]

        # Build parameter options for the selected circuit under this protocol
        available_params = sorted(
            sub.loc[sub["circuit_label"] == selected, "parameter"].dropna().unique()
        )
        if not available_params:
            # Deterministic fallback to the first circuit that has params
            for c in circuits:
                ps = sorted(
                    sub.loc[sub["circuit_label"] == c, "parameter"].dropna().unique()
                )
                if ps:
                    selected = c
                    available_params = ps
                    break
        if not available_params:
            raise ValueError(f"No parameter values found under protocol={protocol} for any circuit (malicious).")

        # Preserve current parameter if still valid; otherwise first available
        param_value = current_param if current_param in available_params else available_params[0]

        return (
            selected,
            [{"label": str(p), "value": p} for p in available_params],
            param_value,
        )

    # Figure + info panel
    @app.callback(
        Output("result-figure", "figure"),
        Output("info-panel", "children"),
        Input("param-dropdown", "value"),
        State("protocol-radio", "value"),
        State("circuit-name", "children"),
    )
    def update_figure(param, protocol, circuit_name):
        fig = build_figure(df, protocol, circuit_name, param)
        cv, bqp, thresh = get_info(circuit_name, circuit_table, protocol)

        return fig, html.Div([
            html.Div(f"Protocol: {protocol}"),
            html.Div(f"BQP error: {bqp:.6f}"),
            html.Div(f"Threshold: {thresh:.6f}"),
            html.Div(f"Correct value: {cv}"),
        ])

    return app


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to your dataset CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    table = load_table()
    app = create_dash_app(df, table)
    app.run(debug=True)


if __name__ == "__main__":
    main()
