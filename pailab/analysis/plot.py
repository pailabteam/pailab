"""This module provides out-of-the box plots to analyse models whee titles, axes labels, additional information is automatically
added to the resuling figures from the information stored in the repository.
"""

import logging
import math
import numpy as np
import pandas as pd
import pailab.analysis.plot_helper as plot_helper  # pylint: disable=E0611
from pailab.ml_repo.repo_store import RepoStore, LAST_VERSION
from pailab.ml_repo.repo import NamingConventions, MLObjectType  # pylint: disable=E0611,E0401
from pailab import RepoInfoKey

has_plotly = True
try:
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    from plotly import tools
    import plotly.graph_objs as go
except ImportError:
    has_plotly = False

logger = logging.getLogger(__name__)

init_notebook_mode(connected=True)


def measure_by_parameter(ml_repo, measure_name, param_name, data_versions=None, training_param=False, logscale_y=False, logscale_x=False):
    """Plot a measure value vs a certain training or model parameter.

    Args:
        ml_repo (MLRepo): MLRepo
        measure_name (str): Name of measure to be plotted.
        param_name (str): Name of parameter to be plotted. To define a subparameter on can use the '/' to define the path to the parameter.
        data_versions (str, optional): Version of the dataset that should be underlying the measure. If Noe, the latest version for the underlying data is used.
                                        Defaults to None.
        training_param (bool, optional): Boolean that defines if parameter of interest belongs to training or model parameter. Defaults to False.
        logscale_y (bool): If true, the y-axis will be log scale. Defaults to False.
        logscale_x (bool): If true, the x-axis will be log scale. Defaults to False.

    Examples:
        To plot the maximum error (which must have been defined in the measures) for the model ``DecisionTreeRegressor`` on the dataset ``sample1`` against
        the parameter ``learning_rate`` contained in the subparameter ``optim_param`` we may call::

            >> measure_by_parameter(ml_repo, 'DecisionTreeRegressor/measure/sample1/max', 'optim_param/learning_rate')
    """
    if logscale_x:
        x_scaler = math.log10
    else:
        x_scaler = lambda x: x

    if logscale_y:
        y_scaler = math.log10
    else:
        y_scaler = lambda x: x

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
                model_label_annotations.append(dict(x=x_scaler(measure[param_name]), y=y_scaler(measure['value']), xref='x', yref='y', text=measure['model_label'],
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

    xaxis = dict(title=param_name)
    if logscale_x:
        xaxis['type'] = 'log'
    yaxis = dict(title=NamingConventions.Measure(
            measure_name).values['measure_type'])
    if logscale_y:
        yaxis['type'] = 'log'
    layout = go.Layout(
        title='measure by parameter',
        annotations=model_label_annotations,
        xaxis=xaxis,
        yaxis=yaxis
    )
    # IPython notebook
    # py.iplot(data, filename='pandas/basic-line-plot')

    fig = go.Figure(data=data, layout=layout)
    # return fig
    iplot(fig)  # , filename='pandas/basic-line-plot')


def projection(ml_repo, left, right, n_steps=100, model=None, labels=None,  output_index=None, direction=None):
    logger.info('Start projection with ' + str(n_steps) + ' steps.')
    x = plot_helper.project(ml_repo, model, labels, left,
                            right, output_index=output_index, n_steps=n_steps)
    # use training data to get output name
    training = ml_repo.get_names(MLObjectType.TRAINING_DATA)
    output_name = ml_repo.get(training[0]).y_coord_names[0]
    data = []
    x_data = [0.0 + float(i)/float(n_steps-1) for i in range(n_steps)]
    for k, v in x.items():
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


def measure_history(ml_repo, measure_name, logscale_y=False):
    """Plots the history of the model w.r.t. a defined measure. The x-axis is defined by the indert datetime of each model.


    Args:
        ml_repo (MLRepo): MLRepo.
        measure_name (str, iterable of str): Name (or iterable of names) of measure(s) to plot (a measure name includes the name of the underlying model and dataset).
        logscale_y (bool): If true, the y-axis will be log scale. Defaults to False.

    Examples:
        To plot the history of the maximum error (which must have been defined in the measures) for the model ``DecisionTreeRegressor`` on the dataset ``sample1``::

            >> measure_history(ml_repo, 'DecisionTreeRegressor/measure/sample1/max')
    """
    if logscale_y:
        y_scaler = math.log10
    else:
        y_scaler = lambda x: x

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
                model_label_annotations.append(dict(x=str(measure['datetime']), y=y_scaler(measure['value']), xref='x', yref='y', text=measure['model_label'],
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

    yaxis = dict(title=NamingConventions.Measure(
            measure_name).values['measure_type'])
    if logscale_y:
        yaxis['type'] = 'log'
    layout = go.Layout(
        title='measure history',
        annotations=model_label_annotations,
        xaxis=dict(title='t'),
        yaxis=yaxis
    )
    # IPython notebook
    # py.iplot(data, filename='pandas/basic-line-plot')
    fig = go.Figure(data=data, layout=layout)

    iplot(fig)  # , filename='pandas/basic-line-plot')


def _histogram(plot_dict, n_bins=None, histnorm='percent'):
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
                                        opacity=opacity,
                                        histnorm=histnorm))
        else:
            plot_data.append(go.Histogram(x=x['x0'],
                                        text=text,
                                        name=k,
                                        opacity=opacity,
                                        nbinsx=n_bins,
                                        histnorm=histnorm))

    fig = go.Figure(data=plot_data, layout=layout)

    iplot(fig)  # , filename='pandas/basic-line-plot')


def histogram_model_error(ml_repo, models, data_name, y_coordinate=None, data_version=LAST_VERSION, n_bins=None,  start_index=0, end_index=-1):
    """Plot histogram of differences between predicted and real values.

    The method plots histograms between predicted and real values of a certain target variable for reference data and models.
    The reference data is described by the data name and the version of the data (as well as the targt variables name). The models can be described
    by
    - a dictionary of model names to versions (a single version number, a range of versions or a list of versions)
    - just a model name (in this case the latest version is used)

    Args:
        ml_repo (MLRepo): [description]
        models (str or dict): A dictionary of model names to versions (a single version number, a range of versions or a list of versions) or
                just a model name (in this case the latest version is used)

        data_name (str or list of str): Name of input data to be used for the error plot.
        y_coordinate (int, str or list, optional): Index or (list of) name(s) of y-coordinate(s) used for error measurement. If None, the first coordinate is used. Defaults to None.
        data_version (str, optional): Version of the input data used. Defaults to LAST_VERSION.

    Examples:
        Plot histograms for errors in the variable mickey_mouse on the dataset my_test_data for the latest version of model_1 and all versions of model_2.

        >>> histogram_model_error(repo, models = {'model_1': ['latest'], 'model_2': ('first','latest')},
            data_name = 'my_test_data', y_coordinate='mickey_mouse')

        Plot histogram for error of latest version of model_2 on the latest version of my_test_data. Note that the plot would be empty if the latest version of model_2
        has not yet been evaluated on the latest version of my_test_data.

        >>> histogram_model_error(repo, models = 'model_2', data_name = 'my_test_data', y_coordinate='mickey_mouse')


    """
    if isinstance(y_coordinate, list):
        plot_dict = {'data':{}, 'x0_name': 'pointwise error'}
        for coord in y_coordinate:
            tmp = plot_helper.get_pointwise_model_errors(
                ml_repo, models, data_name, coord, start_index=start_index, end_index=end_index)
            for k,v in tmp['data'].items():
                plot_dict['data'][str(coord) + ':' + k] = v
        plot_dict['title'] = 'pointwise errors'
    else:    
        plot_dict = plot_helper.get_pointwise_model_errors(
            ml_repo, models, data_name, y_coordinate, start_index=start_index, end_index=end_index)
    _histogram(plot_dict, n_bins)


def scatter_model_error(ml_repo, models, data_name, x_coordinate, y_coordinate=None, start_index=0, end_index=-1):
    '''Plots for each model the pointwise error along a specified target dimension w.r.t. a specified input dimension.

    Args:
        ml_repo (MLRepo): MLRepo.
        models (str, iterable of str or dict of string to string): Definition of the models for whih the pointwise error are plotted.
            It may be a string defining the model or label name (in case of a model the latest model is used), an iterable of strings where again each string is either a model name
            or a label, or a dict mapping model names to version numbers.
        data_name (str or iterable of str): String or iterable of string defining the datasets over which the pointwise errors will be computed.
        x_coordinate (str or int): Either string or int defining the x-coordinate to which the error will be plotted.
        y_coordinate (str or int, optional): Either string or int defining the x-coordinate to which the error will be plotted. Defaults to None (then the 0th cordinate will be used).
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


def histogram_data(ml_repo, data, x_coordinate, n_bins=None,  start_index=0, end_index=-1):
    '''Plot the histogram of the input data along a specified coordinate direction.

    Args:
        ml_repo (MLRepo): MLRepo.
        data (str or dict): Either a string with the name of the data to be plotted (latest data will be plotte) or a dictionary of data names to version or list of versions.
        x_coordinate (str): String defining the x_coordinate to be plotted. Defaults to None.
    '''
    plot_dict = plot_helper.get_data(
        ml_repo, data, x_coordinate, start_index=start_index, end_index=end_index)
    _histogram(plot_dict, n_bins=n_bins)


def histogram_data_conditional_error(ml_repo, models, data, x_coordinate, y_coordinate=0,
                                    start_index=0, end_index=-1, percentile=0.1, n_bins=None, n_hist=1,
                                    metric = 'rbf',  **kwds):
    """Plots the distribution of input data along a given axis for the largest absolute pointwise errors in comparison to the distribution of all data.
    
    Args:
        ml_repo (MLRepo): repository
        models (str or dict): A dictionary of model names (or labels) to versions (a single version number, a range of versions or a list of versions) or 
            just a model name (in this case the latest version is used)
        data (str): name of dataset used for plotting
        x_coordinate (str): Name of x coordinate for which the distribution will be plotted. If None, the method tries to find interesting coordinates where distribution of perentile differs to original distribution.
        y_coordinate (str, optional): Name of y-coordinate for which the error is determined. Defaults to 0 (use first y-coordinate). 
            If None, the method tries to find interesting coordinates where distribution of perentile differs to original distribution.
        start_index (int, optional): Defaults to 0. Startindex of data.
        end_index (int, optional): Defaults to -1. Endindex of data.
        percentile (float, optional): Defaults to 0.1. Percentage of largest absolute errors used.
        n_bins (int, optional): Defaults to None. Number of bin of histogram.
        metric (str or callable, optional): The metric to use when calculating kernel between instances in a feature array. defaults to 'rbf'.
            If metric is a string, it must be one of the metrics in sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS. 
            If metric is precomputed, X is assumed to be a kernel matrix. Alternatively, if metric is a callable function, 
            it is called on each pair of instances (rows) and the resulting value recorded. 
            The callable should take two arrays from X as input and return a value indicating the distance between them. 
            Currently, sklearn provides the following strings: ‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
                                                ‘laplacian’, ‘sigmoid’, ‘cosine’ 
        **kwds: optional keyword parameters that are passed directly to the kernel function.
    
    """
    if x_coordinate is None or y_coordinate is None:
        tmp =  pd.DataFrame.from_dict( plot_helper.get_ptws_error_dist_mmd(ml_repo, models, data, x_coordinate,
                     y_coordinate, start_index=start_index, end_index=end_index, percentile = percentile,  metric='rbf',  **kwds)
            )
        tmp = tmp.sort_values(['mmd'], ascending = False)
        recommended_coordinates = set()
        for i in range(tmp.shape[0]):
            recommended_coordinates.add((tmp.iloc[i]['x-coord'], tmp.iloc[i]['y-coord'], ))
            if len(recommended_coordinates) > n_hist:
                break
        for coord in recommended_coordinates:
            histogram_data_conditional_error(ml_repo, models, data, coord[0], coord[1],
                start_index, end_index, percentile, n_bins, n_hist)
    else:
        tmp = plot_helper.get_pointwise_model_errors(
            ml_repo, models, data, y_coordinate, x_coord_name=x_coordinate, start_index = start_index, end_index = end_index)
        
        plot_data = {}
            
        for k, x in tmp['data'].items():
            abs_err = np.abs(x['x1'])
            sorted_indices = np.argsort(abs_err)
            i_start = int( (1.0-percentile)*len(sorted_indices))
            indices = sorted_indices[i_start:]
            data = {'x0': x['x0'][indices], 'info': x['info']}
            if 'label' in x.keys():
                plot_data[x['label']+':'+str(percentile)] = data
                plot_data[x['label']] = {'x0': x['x0'][start_index:end_index], 'info': x['info']}
            else:
                plot_data[k+':'+str(percentile)] = data
                plot_data[k] = {'x0': x['x0'][start_index:end_index], 'info': x['info']}
        plot_dict = {'data': plot_data, 'title': tmp['title'] + ', inlcuding distribution w.r.t ' + str(100*percentile) + ' % quantile', 'x0_name' : tmp['x0_name'], 'x1_name': tmp['x1_name']}
        _histogram(plot_dict, n_bins=n_bins)


def _ice_plotly(ice_results, ice_points = None, height = None, width = None, ice_results_2 = None, clusters = None):
    data = []
    if ice_points is None:
        ice_points = range(ice_results.ice.shape[0])
    if clusters is not None:
        ice_points_tmp = []
        for i in ice_points:
            if ice_results.labels[i] in clusters:
                ice_points_tmp.append(i)
        _ice_plotly(ice_results, ice_points=ice_points_tmp, ice_results_2 = ice_results_2, height=height, width=width)
        return

    # plot ice curves
    for i in ice_points:
        data.append(go.Scatter(x=ice_results.x_values, y = ice_results.ice[i,:], name = ice_results.data_name + '[' + str(ice_results.start_index +i) + ',:]'))
        if ice_results_2 is not None:
            data.append(go.Scatter(x=ice_results_2.x_values, y = ice_results_2.ice[i,:], name = 'model2, ' + ice_results.data_name + '[' + str(ice_results.start_index +i) + ',:]'))
    layout = go.Layout(
        title='ICE, model: ' + ice_results.model + ', version: ' + ice_results.model_version,
        xaxis=dict(title=ice_results.x_coord_name),
        yaxis=dict(title=ice_results.y_coord_name),
        height = height,
        width = width
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)  # , filename='pandas/basic-line-plot')


def _ice_clusters_plotly(ice_results, height = None, width = None, ice_results_2= None, clusters = None):
    data = []
    if clusters is None:
        clusters = range(ice_results.cluster_centers.shape[0])
    # plot ice cluster curves
    for i in clusters:
        data.append(go.Scatter(x=ice_results.x_values, y = ice_results.cluster_centers[i,:], name = 'cluster ' + str(i)))
    if ice_results_2 is not None:
        cluster_averages = ice_results.compute_cluster_average(ice_results_2)
        for i in clusters:
            data.append(go.Scatter(x=ice_results.x_values, y = cluster_averages[i,:], name = 'average ' + str(i)))

    layout = go.Layout(
        title='ICE clusters, model: ' + ice_results.model + ', version: ' + ice_results.model_version,
        xaxis=dict(title=ice_results.x_coord_name),
        yaxis=dict(title=ice_results.y_coord_name),
        height = height,
        width = width
    )
    # fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Plot 1', 'Plot 2',
    #                                                      'Plot 3', 'Plot 4'))#, layout = layout)
    # fig['layout'].update(title='ICE, model: ' + ice_results.model + ', version: ' + ice_results.model_version, xaxis=dict(title=ice_results.x_coord_name),
    #    yaxis=dict(title=ice_results.y_coord_name),
    #    height = height,
    #    width = width)
    fig = go.Figure(data=data, layout=layout)
    # fig.append_trace(data,1,1)
    iplot(fig)  # , filename='pandas/basic-line-plot')


def ice(ice_results, height = None, width = None, ice_points = None, ice_results_2 = None, clusters = None):
    """Plots ICE graphs computed with tools.interpretation.compute_ice.
    
    Args:
        ice_results (tools.interpretation.ICE_Results): Resulting object after calling tools.interpretation.compute_ice.
        height ([type], optional): [description]. Defaults to None.
        width ([type], optional): [description]. Defaults to None.
        ice_points (list of int, optional): List of integers to select the ICE graphs to be plotted. Defaults to None (all ICE graphs will be plotted).
        ice_results_2 (tools.interpretation.ICE_Results, optional): ICE graphs for another model which are plotted for comparison. Defaults to None.
        clusters (list of int, optional): List of clusters of ICE graphs that will be plotted. In order to wok with this feature, 
                    the ice_result must have been derived calling  tools.interpretation.compute_ice so that clustering will be executed (cluster params need to be set).
                    Defaults to None (no clusters plotted).
    """
    if ice_results_2 is not None:
        ice_results._validate_for_comparison(ice_results_2)
    if has_plotly:
        _ice_plotly(ice_results, height=height, width=width, ice_points = ice_points, ice_results_2 = ice_results_2, clusters = clusters)
    else:
        raise Exception("Plot methods for matplotlib have not yet been implemented.")
     
def ice_clusters(ice_results, height = None, width = None, ice_results_2 = None, clusters = None):
    """Plot the cluster centers from functional clustering of ICE curves.
    
    Args:
        ice_results (ICE_Result): Result from calling the method interpretation.compute_ice with functional clustering.
        height (int, optional): Height of resulting figure. Defaults to None.
        width (int, optional): Width of resulting figures. Defaults to None.
        ice_results_2 (ICE_Result, optional): Result from an ICE computation for a different model (version). 
            Here, the result does not need to contain functional clustering.
            If specified, the ICE curves in this result are clustered according to the clustering of the other results. Defaults to None.
        clusters (iterable of int): Defines the clusters that will be plot. If None, all clusters will be plot. Default to None.

    Raises:
        Exception: If ice_results_2 was computed on a different data (version) or with different start_index, and exception is thrown.
    """
    if ice_results.cluster_centers is None:
        raise Exception('No clusters have yet been computed. Call compute_ice with clusering_param to compute clusters.')

    if has_plotly:
        _ice_clusters_plotly(ice_results, height=height, width=width, ice_results_2=ice_results_2, clusters = clusters)
    else:
        raise Exception("Plot methods for matplotlib have not yet been implemented.")
    
def ice_diff(ice_results, ice_results_2, n_curves=10, ord = 2, height = None, width = None):
    """It computes the indices of the n-th largest ICE differences between two models (w.r.t. a specified vector norm) and plots these ICE graphs.
    
    Args:
        ice_results (tools.interpretation.ICE_Results): [description]
        ice_results_2 (tools.interpretation.ICE_Results): [description]
        n_curves (int, optional): [description]. Defaults to 10.
        ord (int, optional): Defines the norm to be used to measure distance between two ICE curves, see numpy's valid strings for ord in linalg.norm [description]. Defaults to 2.
        
    """
    ice_results._validate_for_comparison(ice_results_2)
    if n_curves > ice_results.ice.shape[0]:
        raise Exception('Number of desired curves exceeds number of all curves.')
    tmp = ice_results.ice - ice_results_2.ice
    distance = np.linalg.norm(ice_results.ice - ice_results_2.ice, ord = ord, axis = 1)
    indices = [i for i in range(ice_results.ice.shape[0])]
    tmp = sorted(zip(distance, indices))
    ice_points = [tmp[i][1] for i in range(n_curves)]
    if has_plotly:
        _ice_plotly(ice_results, height=height, width=width, ice_points = ice_points, ice_results_2 = ice_results_2)
    else:
        raise Exception("Plot methods for matplotlib have not yet been implemented.")
    

