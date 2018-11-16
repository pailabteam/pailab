import logging
import pandas as pd
import pailab.plot_helper as plot_helper  # pylint: disable=E0611
from pailab.repo_store import RepoStore, LAST_VERSION
from pailab.repo import NamingConventions, MLObjectType  # pylint: disable=E0611,E0401
from pailab import RepoInfoKey
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
logger = logging.getLogger(__name__)

init_notebook_mode(connected=True)


def measure_by_model_parameter(ml_repo, measure_name, param_name, data_versions=None):
    x = plot_helper.get_measure_by_model_parameter(
        ml_repo, measure_name, param_name, data_versions)
    data = []
    for k, measures in x.items():
        data_name = str(NamingConventions.Data(
            NamingConventions.EvalData(NamingConventions.Measure(measure_name))))
        data_versions = set()

        model_label_annotations = []
        for measure in measures:
            data_versions.add(measure['data_version'])
            if 'model_label' in measure:
                model_label_annotations.append(dict(x=measure[param_name], y=measure['value'], xref='x', yref='y', text=measure['model_label'],
                                                    showarrow=True,
                                                    arrowhead=7,
                                                    ax=0,
                                                    ay=-20))
        measures = pd.DataFrame(measures)
        for d_version in data_versions:
            # if True:
            df = measures.loc[measures['data_version'] == d_version]
            text = ["model version: " + str(x['model_version']) + '<br>' +
                    data_name + ': ' + str(x['data_version']) + '<br>'
                    + 'train_data: ' + str(x['train_data_version'])
                    for index, x in df.iterrows()]

            if True:  # len(x) > 1:
                plot_name = k + ': ' + str(d_version)
            # else:
            #    plot_name = data_name + ': ' + str(d_version)
            data.append(
                go.Scatter(
                    x=df[param_name],
                    y=df['value'],
                    text=text,
                    name=plot_name,
                    mode='markers'
                )

            )

    layout = go.Layout(
        title='measure by parameter',
        annotations=model_label_annotations,
        xaxis=dict(title=param_name),
        yaxis=dict(title=NamingConventions.Measure(
            measure_name).values['measure_type'])
    )
    # IPython notebook
    # py.iplot(data, filename='pandas/basic-line-plot')
    fig = go.Figure(data=data, layout=layout)

    iplot(fig)  # , filename='pandas/basic-line-plot')


def _histogram(plot_dict):
    layout = go.Layout(
        title=plot_dict['title'],
        xaxis=dict(title=plot_dict['x0_name']),
        barmode='overlay'
    )
    plot_data = []
    opacity = 1.0
    if len(plot_dict['data'].keys()) > 1:
        opacity = 0.75
    for k, x in plot_dict['data'].items():
        text = ''
        for l, w in x['info'].items():
            text += l + ':' + str(w) + '<br>'
        if 'label' in x.keys():
            k = k + ', ' + x['label']
        plot_data.append(go.Histogram(x=x['x0'],
                                      text=text,
                                      name=k,
                                      opacity=opacity))
    fig = go.Figure(data=plot_data, layout=layout)

    iplot(fig)  # , filename='pandas/basic-line-plot')


def histogram_model_error(ml_repo, models, data_name, y_coordinate=None, data_version=LAST_VERSION):
    plot_dict = plot_helper.get_pointwise_model_errors(
        ml_repo, models, data_name, y_coordinate)
    _histogram(plot_dict)


def scatter_model_error(ml_repo, models, data_name, x_coordinate, y_coordinate=None, data_version=LAST_VERSION):
    plot_dict = plot_helper.get_pointwise_model_errors(
        ml_repo, models, data_name, y_coordinate, x_coord_name=x_coordinate)

    layout = go.Layout(
        title=plot_dict['title'],
        xaxis=dict(title=plot_dict['x0_name']),
        yaxis=dict(title=plot_dict['x1_name'])
    )
    plot_data = []
    for k, x in plot_dict['data'].items():
        text = ''
        for l, w in x['info'].items():
            text += l + ':' + str(w) + '<br>'
        if 'label' in x.keys():
            k = k + ', ' + x['label']
        plot_data.append(go.Scatter(x=x['x0'],
                                    y=x['x1'],
                                    text=text,
                                    name=k,
                                    mode='markers'))

    # IPython notebook
    # py.iplot(data, filename='pandas/basic-line-plot')
    fig = go.Figure(data=plot_data, layout=layout)

    iplot(fig)  # , filename='pandas/basic-line-plot')


def histogram(ml_repo, data, x_coordinate=None, y_coordinate=None):
    if x_coordinate is None and y_coordinate is None:
        raise Exception(
            "Please specify either an x- or a y-coordinate to plot.")
    data_names = data
    if isinstance(data, str):
        data_names = [data]
    plot_data = []
    for d in data_names:
        # if just string, get lates version, otherwise assume it is list or tuple with version info
        name = d
        version = RepoStore.LAST_VERSION
        if not isinstance(d, str):
            name = d[0]
            version = d[1]
        tmp = ml_repo.get(name, version=version, full_object=True)
        if x_coordinate is not None:
            i = tmp.x_coord_names.index(x_coordinate)
            data = tmp.x_data
            coord_name = x_coordinate
        else:
            i = tmp.y_coord_names.index(y_coordinate)
            coord_name = y_coordinate
            data = tmp.y_data

        plot_data.append(go.Histogram(x=data[:, i]))

    layout = go.Layout(
        title='histogram ' + coord_name,
        xaxis=dict(title=coord_name),
    )
    fig = go.Figure(data=plot_data, layout=layout)

    iplot(fig)  # , filename='pandas/basic-line-plot')
