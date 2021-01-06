from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd


def component_heatmap(df):
    components = df['component'].unique()

    fig = make_subplots(rows=1, cols=len(components), subplot_titles=components)
    for c, component in enumerate(components):
        current = df[df['component'] == component]
        current = current.pivot(index='layer', columns='lang', values='score')
        fig.add_trace(go.Heatmap(z=current, x=current.columns, y=current.index,
                                 zmin=0, zmax=100, showlegend=not c, colorscale='RdBu'),
                      row=1, col=c+1)

    fig.show()


def mlp_heatmap(df):
    steps = [19]
    df['step'] = pd.to_numeric(df['step'])

    fig = make_subplots(rows=1, cols=len(steps), subplot_titles=steps)
    for n, step in enumerate(steps):
        current = df[df['step'] == step]
        current = current.pivot(index='layer', columns='lang', values='score')
        fig.add_trace(go.Heatmap(z=current, x=current.columns, y=current.index,
                                 zmin=0, zmax=100, showlegend=not n, colorscale='RdBu'),
                      row=1, col=n+1)

    fig.show()
