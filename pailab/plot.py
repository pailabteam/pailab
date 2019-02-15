"""This module provides out-of-the box plots to analyse models whee titles, axes labels, additional information is automatically 
added to the resuling figures from the information stored in the repository. 
"""

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


def measure_by_parameter(ml_repo, measure_name, param_name, data_versions=None, training_param=False):
    '''[summary]

    Args:
        :param ml_repo ([type]): [description]
        :param measure_name ([type]): [description]
        :param param_name ([type]): [description]
        :param data_versions ([type], optional): Defaults to None. [description]
        :param training_parm (bool, optional): If True, training parameters are used otherwise model parameter (default is False)
    '''

    x = plot_helper.get_measure_by_parameter(
        ml_repo, measure_name, param_name, data_versions, training_param)
    data = []
    model_label_annotations = []
    for k, measures in x.items():
        data_name = str(NamingConventions.Data(
            NamingConventions.EvalData(NamingConventions.Measure(measure_name))))
        data_versions = set()

        for measure in measures:
            data_versions.add(measure['data_version'])
            if 'model_label' in measure:
                model_label_annotations.append(dict(x=measure[param_name], y=measure['value'], xref='x', yref='y', text=measure['model_label'],
                                                    showarrow=True,
                                                    arrowhead=2,
                                                    # ax=0,
                                                    # ay=-30
                                                    ))
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
    #return fig
    iplot(fig)  # , filename='pandas/basic-line-plot')


def projection(ml_repo, left, right, n_steps = 100, model = None, labels = None,  output_index = None, direction = None):
    logger.info('Start projection with ' + str(n_steps) + ' steps.')
    x = plot_helper.project(ml_repo, model, labels, left, right, output_index=output_index, n_steps= n_steps)
    training = ml_repo.get_names(MLObjectType.TRAINING_DATA) #use training data to get output name
    output_name = ml_repo.get(training[0]).y_coord_names[0]
    data = []
    x_data = [0.0 + float(i)/float(n_steps-1) for i in range(n_steps) ]
    for k,v in x.items():
        data.append(
                    go.Scatter(
                        x=x_data,
                        y=v,
                        name=k
                    )
        )
    layout = go.Layout(
        title='projection',
        xaxis=dict(title='steps'),
        yaxis=dict(title=output_name)
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

def measure_history(ml_repo, measure_name):

    x = plot_helper.get_measure_history(
        ml_repo, measure_name)
    data = []
    model_label_annotations = []
    for k, measures in x.items():
        data_name = str(NamingConventions.Data(
            NamingConventions.EvalData(NamingConventions.Measure(measure_name))))
        data_versions = set()

        for measure in measures:
            data_versions.add(measure['data_version'])
            if 'model_label' in measure:
                model_label_annotations.append(dict(x=str(measure['datetime']), y=measure['value'], xref='x', yref='y', text=measure['model_label'],
                                                    showarrow=True,
                                                    arrowhead=2,  # 1
                                                    # ax=,
                                                    # ay=-30
                                                    ))
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
                    x=df['datetime'],
                    y=df['value'],
                    text=text,
                    name=plot_name,
                    mode='markers'
                )

            )

    layout = go.Layout(
        title='measure history',
        annotations=model_label_annotations,
        xaxis=dict(title='t'),
        yaxis=dict(title=NamingConventions.Measure(
            measure_name).values['measure_type'])
    )
    # IPython notebook
    # py.iplot(data, filename='pandas/basic-line-plot')
    fig = go.Figure(data=data, layout=layout)

    iplot(fig)  # , filename='pandas/basic-line-plot')


def _histogram(plot_dict, n_bins = None):
    layout = go.Layout(
        title=plot_dict['title'],
        xaxis=dict(title=plot_dict['x0_name']),
        barmode='overlay'
    )
    plot_data = []
    opacity = 1.0
    if len(plot_dict['data'].keys()) > 1:
        opacity = 0.5
    for k, x in plot_dict['data'].items():
        text = ''
        for l, w in x['info'].items():
            text += l + ':' + str(w) + '<br>'
        if 'label' in x.keys():
            k = x['label'] + ', ' + k
        if n_bins is None:
            plot_data.append(go.Histogram(x=x['x0'],
                                        text=text,
                                        name=k,
                                        opacity=opacity))
        else:
            plot_data.append(go.Histogram(x=x['x0'],
                                        text=text,
                                        name=k,
                                        opacity=opacity, 
                                        nbinsx = n_bins))
    fig = go.Figure(data=plot_data, layout=layout)

    iplot(fig)  # , filename='pandas/basic-line-plot')


def histogram_model_error(ml_repo, models, data_name, y_coordinate=None, data_version=LAST_VERSION, n_bins = None,  start_index = 0, end_index = -1):
    """Plot histogram of differences between predicted and real values.

    The method plots histograms between predicted and real values of a certain target variable for reference data and models. 
    The reference data is described by the data name and the version of the data (as well as the targt variables name). The models can be described
    by 
        - a dictionary of model names to versions (a single verion numbr, a range of versions or a list of versions)
        - just a model name (in this case the latest version is used)

    Args:
        :param ml_repo ([type]): [description]
        :param models ([type]): [description]
        :param data_name (str or list of str): [description]
        :param y_coordinate ([type], optional): Defaults to None. [description]
        :param data_version ([type], optional): Defaults to LAST_VERSION. [description]

    Examples:
        Plot histograms for errors in the variable mickey_mouse on the dataset my_test_data for the latest version of model_1 and all versions of model_2. 

        >>> histogram_model_error(repo, models = {'model_1': ['latest'], 'model_2': ('first','latest')}, 
            data_name = 'my_test_data', y_coordinate='mickey_mouse')

        Plot histogram for error of latest version of model_2 on the latest version of my_test_data. Note that the plot would be empty if the latest version of model_2
        has not yet been evaluated on the latest version of my_test_data.

        >>> histogram_model_error(repo, models = 'model_2', data_name = 'my_test_data', y_coordinate='mickey_mouse')


    """

    plot_dict = plot_helper.get_pointwise_model_errors(
        ml_repo, models, data_name, y_coordinate, start_index=start_index, end_index=end_index)
    _histogram(plot_dict, n_bins)


def scatter_model_error(ml_repo, models, data_name, x_coordinate, y_coordinate=None, data_version=LAST_VERSION,  start_index = 0, end_index = -1):
    '''[summary]

    Args:
        :param ml_repo ([type]): [description]
        :param models ([type]): [description]
        :param data_name ([type]): [description]
        :param x_coordinate ([type]): [description]
        :param y_coordinate ([type], optional): Defaults to None. [description]
        :param data_version ([type], optional): Defaults to LAST_VERSION. [description]
    '''

    plot_dict = plot_helper.get_pointwise_model_errors(
        ml_repo, models, data_name, y_coordinate, x_coord_name=x_coordinate, start_index=start_index, end_index=end_index)

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
            k = x['label'] + ', ' + k
        plot_data.append(go.Scatter(x=x['x0'],
                                    y=x['x1'],
                                    text=text,
                                    name=k,
                                    mode='markers'))

    # IPython notebook
    # py.iplot(data, filename='pandas/basic-line-plot')
    fig = go.Figure(data=plot_data, layout=layout)

    iplot(fig)  # , filename='pandas/basic-line-plot')


def histogram_data(ml_repo, data, x_coordinate, y_coordinate=None):
    '''[summary]

    Args:
        ml_repo ([type]): [description]
        data ([type]): [description]
        x_coordinate ([type]): Defaults to None. [description]
        y_coordinate ([type], optional): Defaults to None. [description]

    Raises:
        Exception: [description]
    '''
    plot_dict = plot_helper.get_data(ml_repo, data, x_coordinate)
    _histogram(plot_dict)
