from plotly.subplots import make_subplots
import plotly.graph_objects as go


def component_heatmap(df):
    steps = [0, 10, 19]
    components = df['component'].unique()

    fig = make_subplots(rows=len(components), cols=3, subplot_titles=steps)
    for r, component in components:
        for c, step in enumerate(steps):
            current = df[df['step'] == step][df['component'] == component]
            current = current.pivot(index='layer', columns='lang', values='score')
            fig.add_trace(go.Heatmap(z=current, x=current.columns, y=current.index,
                                     zmin=0, zmax=100, showlegend=not (r and c), colorscale='RdBu'),
                          row=r+1, col=c+1)

    fig.show()


def mlp_heatmap(df):
    steps = [0, 10, 19]

    fig = make_subplots(rows=1, cols=3, subplot_titles=steps)
    for n, step in enumerate(steps):
        current = df[df['step'] == step]
        current = current.pivot(index='layer', columns='lang', values='score')
        fig.add_trace(go.Heatmap(z=current, x=current.columns, y=current.index,
                                 zmin=0, zmax=100, showlegend=not n, colorscale='RdBu'),
                      row=1, col=n+1)

    fig.show()

