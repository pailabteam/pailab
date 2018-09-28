import logging
import pandas as pd
import repo.plot_helper as plot_helper
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
logger = logging.getLogger('repo.plot')

init_notebook_mode(connected=True)


def measure_by_model_parameter(ml_repo, measure_name, param_name):
    measures = plot_helper.get_measure_by_model_parameter(
        ml_repo, measure_name, param_name)
    hover_text = ["model version: " +
                  str(x['model_version']) for x in measures]
    measures = pd.DataFrame(measures)

    data = [
        go.Scatter(
            x=[i for i in range(len(measures.index))],
            y=measures['value'],
            text=hover_text,
            name=measure_name
        )
    ]

    layout = go.Layout(
        title='measure by parameter',
        xaxis=dict(title=param_name),
        yaxis=dict(title=measure_name)
    )
    # IPython notebook
    # py.iplot(data, filename='pandas/basic-line-plot')
    fig = go.Figure(data=data, layout=layout)

    url = iplot(fig, filename='pandas/basic-line-plot')
