import logging
import pandas as pd
import repo.plot_helper as plot_helper
from repo.repo import NamingConventions, MLObjectType
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
logger = logging.getLogger('repo.plot')

init_notebook_mode(connected=True)


def measure_by_model_parameter(ml_repo, measure_name, param_name, data_versions=None):
    measures = plot_helper.get_measure_by_model_parameter(
        ml_repo, measure_name, param_name, data_versions)
    data_name = str(NamingConventions.Data(
        NamingConventions.EvalData(NamingConventions.Measure(measure_name))))
    data_versions = set()
    for measure in measures:
        data_versions.add(measure['data_version'])
    measures = pd.DataFrame(measures)
    data = []

    for d_version in data_versions:
        # if True:
        df = measures.loc[measures['data_version'] == d_version]
        text = ["model version: " + str(x['model_version']) + '<br>' +
                data_name + ': ' + str(x['data_version']) + '<br>'
                + 'train_data: ' + str(x['train_data_version'])
                for index, x in df.iterrows()]
        data.append(
            go.Scatter(
                x=df[param_name],
                y=df['value'],
                text=text,
                name=data_name + ': ' + str(d_version),
                mode='markers'
            )

        )

    layout = go.Layout(
        title='measure by parameter',
        xaxis=dict(title=param_name),
        yaxis=dict(title=NamingConventions.Measure(
            measure_name).values['measure_type'])
    )
    # IPython notebook
    # py.iplot(data, filename='pandas/basic-line-plot')
    fig = go.Figure(data=data, layout=layout)

    url = iplot(fig, filename='pandas/basic-line-plot')
