import pailab.analysis.plot as plot
import pailab.analysis.plot_helper as plt_helper
from ipywidgets import *
import ipywidgets as widgets
import numpy as np
import copy
from IPython.display import display, clear_output

from pailab import MLObjectType, RepoInfoKey, FIRST_VERSION, LAST_VERSION
import pailab.tools.checker as checker
import pailab.tools.tools as tools
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# set option so that long lines have a linebreak
pd.set_option('display.max_colwidth', -1)

beakerX = False
if beakerX:
    from beakerx import TableDisplay
    #from beakerx.object import beakerx
else:
    def TableDisplay(dt):
        display(dt)


class _MLRepoModel:

    class _DataModel:
        def __init__(self, ml_repo):
            self._training_data = {}
            self._test_data = {}
            for k in ml_repo.get_names(MLObjectType.TRAINING_DATA):
                tmp = ml_repo.get(k)
                self._training_data[k] = tmp.n_data
                self._x_coord_names = tmp.x_coord_names
                self._y_coord_names = tmp.y_coord_names
            for k in ml_repo.get_names(MLObjectType.TEST_DATA):
                tmp = ml_repo.get(k)
                self._test_data[k] = tmp.n_data

        def get_data_names(self):
            result = [k for k in self._test_data.keys()]
            result.extend([k for k in self._training_data.keys()])
            return result

        def get_num_data(self, data):
            result = []
            for d in data:
                if d in self._test_data.keys():
                    result.append(self._test_data[d])
                elif d in self._training_data.keys():
                    result.append(self._training_data[d])
                else:
                    raise Exception('Cannot find data ' + d)
            return result

    class _ModelModel:
        def __init__(self, ml_repo):
            self._model_info_table = self._setup_model_info_table(ml_repo)

        def _setup_model_info_table(self, ml_repo):
            model_rows = []
            model_names = ml_repo.get_names(MLObjectType.CALIBRATED_MODEL)
            for model_name in model_names:
                models = ml_repo.get(model_name, version=(
                    FIRST_VERSION, LAST_VERSION), full_object=False)
                for model in models:
                    tmp = copy.deepcopy(model.repo_info.get_dictionary())
                    tmp['model'] = tmp['name']
                    del tmp['big_objects']
                    del tmp['modifiers']
                    del tmp['modification_info']
                    model_rows.append(tmp)
            model_info_table = pd.DataFrame(model_rows)
            model_info_table.set_index(['model', 'version'], inplace=True)
            return model_info_table

        def get_info_table(self):
            return self._model_info_table

        def setup_error_measure_table(self, ml_repo, data_sets, measures):
            tmp = []
            for measure in measures:
                for data in data_sets:
                    tmp.append(pd.DataFrame(
                        tools.get_model_measure_list(ml_repo,  measure, data)))
                    tmp[-1].set_index(['model', 'version'], inplace=True)
            result = self.get_info_table()
            tmp.insert(0, result)
            return pd.concat(tmp, axis=1)

    class _ConsistencyModel:
        def __init__(self, ml_repo):
            self.tests = checker.Tests.run(ml_repo)
            self.model = checker.Model.run(ml_repo)
            self.data = checker.Data.run(ml_repo)

    def __init__(self):
        pass

    def set_repo(self, ml_repo):
        self.ml_repo = ml_repo
        self._setup()

    def _setup(self):
        self.object_types = {}
        for k in MLObjectType:
            self.object_types[k.value] = self.ml_repo.get_names(k)
        self.data = _MLRepoModel._DataModel(self.ml_repo)
        self.model = _MLRepoModel._ModelModel(self.ml_repo)
        self.consistency = _MLRepoModel._ConsistencyModel(self.ml_repo)
        self._setup_measures()
        self._setup_labels()

    def _setup_labels(self):
        self.labels = {}
        label_names = self.ml_repo.get_names(MLObjectType.LABEL)
        if label_names is None:
            return
        if isinstance(label_names, str):
            label_names = [label_names]
        for l in label_names:
            label = self.ml_repo.get(l)
            self.labels[l] = {'model': label.name, 'version': label.version}
        

    def _setup_measures(self):
        measure_names = self.ml_repo.get_names(
            MLObjectType.MEASURE_CONFIGURATION)
        if len(measure_names) == 0:
            self.measures = []
        else:
            measure_config = self.ml_repo.get(measure_names[0])
            self.measures = [x for x in measure_config.measures.keys()]

    def get_model_statistics(self):
        model_stats = {}
        models = self.ml_repo.get_names(MLObjectType.CALIBRATED_MODEL)
        if isinstance(models, str):
            models = [models]
        for m in models:
            model = self.ml_repo.get(m)
            model_stats[model.repo_info.name] = {
                'last commit': model.repo_info.commit_date,
                '#total commits': self.model.get_info_table().shape[0]
            }
        return model_stats
        


widget_repo = _MLRepoModel()

# region helpers


def _add_title_and_border(name):
    def _get_widget(get_widget):
        def wrapper(self):
            return widgets.VBox(children=[
                widgets.HTML(value='<h3 style="Color: white; background-color:#d1d1e0; text-align: center"> ' + name + '</h3>'),#, layout = widgets.Layout(width = '100%')),
                get_widget(self),
                widgets.HTML(value='<h3 style="Color: white; background-color:#d1d1e0; text-align: center"> </h3>')#, layout = widgets.Layout(width = '100%'))
            ], layout=widgets.Layout(padding='0px 0px 0px 0px', overflow_x='auto')#, overflow_y='auto', ) 
            ) #layout=widgets.Layout(border='solid 1px'))
        return wrapper
    return _get_widget


def _highlight_max(data, color='red'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'color: {}'.format(color)
    # remove % and cast to float
    #data = data.replace('%','', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


def _highlight_min(data, color='green'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'color: {}'.format(color)
    # remove % and cast to float
    #data = data.replace('%','', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.min()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.min().min()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


class _TableViewer:
    def __init__(self, table, table_name, selected_columns=None):
        self._table = table
        self._table_name = table_name
        self._columns = table.columns
        if selected_columns is None:
            self._selected_columns = self._columns
        else:
            self._selected_columns = selected_columns

        self._selected_columns = widgets.SelectMultiple(
            options=self._columns, value=self._selected_columns)
        self._output = widgets.Output()

        self._settings = widgets.HBox(children=[])
        self._tab = widgets.Tab(children=[self._output, self._settings], title=[
                                'Table', 'Table Settings'])

        self._button_update = widgets.Button(description='update')
        self._button_update.on_click(self.get_overview)

    def get_overview(self, d):
        with self._output:
            clear_output(wait=True)
            # , orient='index'))
            TableDisplay(self._table[self._selected_columns.value])

    def get_widget(self):
        return self._tab


class _ObjectCategorySelector:

    def __init__(self, *args, **kwargs):
        selection = []
        for k, v in widget_repo.object_types.items():
            if len(v) > 0:
                selection.append(k + ' (' + str(len(v)) + ')')
        if 'layout' not in kwargs.keys():
            kwargs['layout'] = widgets.Layout(width='300px', height='250px')
        kwargs['value'] = []
        self._selector = widgets.SelectMultiple(options=selection, **kwargs)

    def get_selection(self):
        return [k.split(' ')[0] for k in self._selector.value]

    def get_widget(self):
        return widgets.VBox(children=[
                            widgets.Label(value='Object Types'),
                            self._selector
                            ]
                            )


class _DataSelector:
    """Widget to select training and test data.
    """

    def __init__(self, **kwargs):
        self._selection_widget = widgets.SelectMultiple(
            options=widget_repo.data.get_data_names(), **kwargs)

    def get_widget(self):
        return widgets.VBox(children=[widgets.Label(value='Data'), self._selection_widget])

    def get_selection(self):
        return self._selection_widget.value


class _MeasureSelector:
    """Widget to select training and test data.
    """

    def __init__(self, **kwargs):
        self._selection_widget = widgets.SelectMultiple(
            options=widget_repo.measures, **kwargs)

    def get_widget(self):
        return widgets.VBox(children=[widgets.Label(value='Measures'), self._selection_widget])

    def get_selection(self):
        return self._selection_widget.value

# endregion


class ObjectOverviewList:
    def __init__(self, beakerX=False):
        self._categories = _ObjectCategorySelector(
            layout=widgets.Layout(width='250px', height='250px'))
        self._repo_info = widgets.SelectMultiple(
            options=[k.value for k in RepoInfoKey], value=['category', 'name', 'commit_date', 'version'], 
            layout=widgets.Layout(width='200px', height='250px', margin = '10px')
        )
        #self._settings = widgets.HBox(children=[self.categories, self._repo_info])

        self._button_update = widgets.Button(description='update')
        self._button_update.on_click(self.get_overview)

        self._output = widgets.Output(layout = widgets.Layout(height = '300px',width = '1000px', overflow_y='auto',  overflow_x='auto') )
        self._input_box = widgets.HBox(
            children=[
                self._categories.get_widget(),
                widgets.VBox(children=[
                    widgets.Label(value='Info Fields'),
                    self._repo_info
                ]
                ),
                widgets.VBox(children=[
                    self._button_update,
                    self._output
                    ],
                    layout = widgets.Layout(margin = '10px 10px 10px 10px')
                )
            ]
        )

    def get_overview(self, d):
        result = {}
        for info in self._repo_info.value:
            result[info] = []

        for k in self._categories.get_selection():
            for n in widget_repo.object_types[k]:
                obj = widget_repo.ml_repo.get(n)
                for info in self._repo_info.value:
                    if isinstance(obj.repo_info[info], MLObjectType):
                        result[info].append(obj.repo_info[info].value)
                    else:
                        result[info].append(str(obj.repo_info[info]))
        with self._output:
            clear_output(wait=True)
            TableDisplay(pd.DataFrame.from_dict(result))  # , orient='index'))

    @_add_title_and_border('Object Overview')
    def get_widget(self):
        return self._input_box


class ObjectView:

    def _setup_names(self, change=None):
        names = []
        for k in self._categories.get_selection():
            names.extend(widget_repo.ml_repo.get_names(k))
        self._names.options = names

    def __init__(self):
        self._categories = _ObjectCategorySelector()
        self._names = widgets.SelectMultiple(
            options=[]
        )
        self._setup_names()
        self._categories.observe(self._setup_names, 'value')

        self._button_update = widgets.Button(description='show history')
        self._button_update.on_click(self.show_history)
        self._output = widgets.Output()
        self._input_box = widgets.HBox(
            children=[self._categories.get_widget(), self._names, self._button_update, self._output], layout=widgets.Layout(border='solid 1px')
        )

    def show_history(self, d):
        result = {RepoInfoKey.NAME.value: [],
                  RepoInfoKey.AUTHOR.value: [],
                  RepoInfoKey.VERSION.value: [],
                  RepoInfoKey.COMMIT_DATE.value: []}
        for k in self._names.value:
            history = widget_repo.ml_repo.get_history(k)
            for l in history:
                for m in result.keys():
                    result[m].append(l['repo_info'][m])
        with self._output:
            clear_output(wait=True)
            TableDisplay(pd.DataFrame.from_dict(result))

    @_add_title_and_border('Object View')
    def get_widget(self):
        return self._input_box


class RepoOverview:
    def __init__(self):
        self._repo_name = widgets.HTML(
            value='<div style="background-color:#c2c2d6"><h4 stype="text-align: center"> Repository: ' 
                + widget_repo.ml_repo._config['name'] + '</h4>')#, margin = '0px 0px 0px 0px'))
        self._data_statistics = widgets.Output(
            layout=Layout(width='450px', height='450px'))
        self._plot_data_statistics()
        self._measures = widgets.Output(
            layout=Layout(width='450px', height='450px'))
        self._consistency = self._setup_consistency()
        self._labels = self._setup_labels()
        self._model_stats = self._setup_model_stats()

    # check consistency
       
    def _setup_consistency(self):
        def create_consistency_html(**kwargs):
            result = '<div style="background-color:#c2c2d6">'
            result += '<h4 stype="text-align: center">Consistency</h4>'
            for k,v in kwargs.items():
                if len(v) > 0:
                    result += '<p style="background-color:red"> ' + str(v) + ' ' + k +' issues found!</p>'
                else:
                    result += '<p style="background-color:lightgreen">No ' + k + ' issues found.</p>'
            result += '</div>'
            return result
        
        return widgets.HTML(create_consistency_html(model = widget_repo.consistency.model, 
                                test = widget_repo.consistency.tests, 
                                data = widget_repo.consistency.data), 
                            layout = widgets.Layout(margin = '0% 0% 0% 0%', width='400px'))

    def _setup_labels(self):
        header = widgets.HTML('<div style="background-color:#c2c2d6"><h4 stype="text-align: center">Labels</h4>')
        label_output = None
            
        if len(widget_repo.labels)>0:
            label_output = widgets.Output(
                layout=Layout(width='400px', height='100px', overflow_y='auto', overflow_x='auto'))
            with label_output:
                clear_output(wait=True)
                display(pd.DataFrame.from_dict(widget_repo.labels, orient='index'))
        else:
            label_output = widgets.HTML('<div style="background-color:#ff4d4d"><h4 stype="text-align: center">No labels defined.</h4>')

        return widgets.VBox(children=[header, label_output])

    def _setup_model_stats(self):
        header = widgets.HTML('<div style="background-color:#c2c2d6"><h4 stype="text-align: center">Models</h4>')
        model_stats_output = widgets.Output(
                layout=Layout(width='400px', height='100px', overflow_y='auto', overflow_x='auto'))
        with model_stats_output:
            clear_output(wait=True)
            display(pd.DataFrame.from_dict(widget_repo.get_model_statistics(), orient='index'))
        return widgets.VBox(children=[header, model_stats_output])

    def _plot_data_statistics(self):
        data_names = widget_repo.data.get_data_names()
        data_num_points = widget_repo.data.get_num_data(data_names)
        with self._data_statistics:
            clear_output(wait=True)
            plt.rcdefaults()
            _, ax = plt.subplots()
            y_pos = np.arange(len(data_names))
            ax.barh(y_pos, data_num_points, align='center',
                    color='green', ecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(data_names, rotation=45, va='top')
            ax.invert_yaxis()
            ax.set_xlabel('number of datapoints')
            ax.set_title('Datasets')
            plt.show()

    def _plot_measures(self):
        with self._measures:
            clear_output(wait=True)
            plt.rcdefaults()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for data in widget_repo.data.get_data_names():
                for measure in widget_repo.measures:
                    error = widget_repo.model.setup_error_measure_table(
                        widget_repo.ml_repo, [data], [measure])
                    error = error.sort_values(by='commit_date')
                    plt.plot(error['commit_date'],
                            error[measure + ', ' + data], '-x', label=measure + ', ' + data)
            plt.xlabel('commit date')
            ax.grid(True)
            ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
            for label in ax.get_xticklabels():
                label.set_rotation(40)
                label.set_horizontalalignment('right')
            #fig.autofmt_xdate()
            plt.legend()
            ax.set_title('Measures')
            #plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
            plt.show()
            # plt.set_title('Measures')

    @_add_title_and_border('Repository Overview')
    def get_widget(self):
        self._plot_measures()
        return widgets.HBox(children=[
                    widgets.VBox(children=[self._repo_name, self._model_stats, self._labels, self._consistency], 
                        layout=widgets.Layout(width='400px')),
                    widgets.VBox(children=[self._measures]),
                    widgets.VBox(children=[self._data_statistics])
                    ],
                    layout=widgets.Layout(width='100%', height='100%')
                )


class MeasureView:
    def __init__(self, beakerX=False):
        self._data = _DataSelector()
        self._measures = _MeasureSelector()
        self._repo_info = widgets.SelectMultiple(
            options=[k.value for k in RepoInfoKey], value=['category', 'name', 'commit_date', 'version'], layout=widgets.Layout(width='200px', height='250px')
        )
        self._output = widgets.Output(layout=widgets.Layout(
            width='1000px', height='450px', overflow_y='auto', overflow_x='auto'))
        self._button_update = widgets.Button(description='update')
        self._button_update.on_click(self.get_measures)

    def _get_columns_selected(self):
        columns = [x for x in self._repo_info.value]
        for data in self._data.get_selection():
            for m in self._measures.get_selection():
                columns.append(m+', '+data)
        return columns

    @_add_title_and_border('Measure View')
    def get_widget(self):
        self._tab = widgets.Tab(children=[
            self._output,
            widgets.HBox(children=[
                self._data.get_widget(),
                self._measures.get_widget(),
                widgets.VBox(children=[
                    widgets.Label(
                        value='Model Columns'),
                    self._repo_info]
                ),
                self._button_update
            ])
        ],
            title=['Table', 'Settings']
        )
        self._tab.set_title(0, 'Table')
        self._tab.set_title(1, 'Settings')
        return self._tab

    def get_measures(self, d):
        self._tab.selected_index = 0
        tmp = widget_repo.model.setup_error_measure_table(
            widget_repo.ml_repo, self._data.get_selection(), self._measures.get_selection())
        columns = [c for c in tmp.columns if c in self._get_columns_selected()]
        tmp2 = tmp[columns]
        with self._output:
            clear_output(wait=True)
            # apply highlighting to floating columns only
            floats = [x.kind == 'f' for x in tmp2.dtypes]
            float_columns = tmp2.columns[floats]
            TableDisplay(tmp2.style.apply(_highlight_max, subset=float_columns).apply(
                _highlight_min, subset=float_columns))  # , orient='index'))


class ConsistencyChecker:

    def _consistency_check(self):
        self._test_results = checker.Tests.run(self._ml_repo)
        self._model_results = checker.Model.run(self._ml_repo)
        self._data_results = checker.Data.run(self._ml_repo)

    def __init__(self, ml_repo, beakerX=False):
        self._ml_repo = ml_repo

        self._overview_output = widgets.Output()

        self._button_update = widgets.Button(description='update')
        self._button_update.on_click(self.show_checks)

        self._widget_main = widgets.VBox(
            children=[self._button_update, self._overview_output]
        )

    def show_checks(self, d):
        self._consistency_check()
        with self._overview_output:
            clear_output(wait=True)
            print('test issues: ' + str(len(self._test_results)) + ', model issues: ' +
                  str(len(self._model_results)) + ', data issues: ' + str(len(self._data_results)))
            result = {'test': [], 'message': [], 'model': [], 'type': []}
            for k, v in self._test_results.items():
                for a, b in v.items():
                    result['type'].append('test')
                    result['model'].append(k)
                    result['test'].append(a)
                    result['message'].append(b)
            for k, v in self._model_results.items():
                for a, b in v.items():
                    result['type'].append('model')
                    result['model'].append(k)
                    result['test'].append(a)
                    result['message'].append(b)
            for k, v in self._data_results.items():
                for a, b in v.items():
                    result['type'].append('data')
                    result['model'].append(k)
                    result['test'].append(a)
                    result['message'].append(b)

            display(pd.DataFrame.from_dict(result))

    @_add_title_and_border('Consistency')
    def get_widget(self):
        return self._widget_main


# import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, iplot


class Plotter:
    def _get_measures(self):
        measures = self._ml_repo.get_names(MLObjectType.MEASURE)
        result = []
        for m in measures:
            if self._model_selection.value in m:
                result.append(m)
        self._data_selection.options = result

    def _get_parameter(self, dummy=None):
        obj_cat = MLObjectType.TRAINING_PARAM
        if self._train_model_sel.value == 'model':
            obj_cat = MLObjectType.MODEL_PARAM
        parameters = self._ml_repo.get_names(obj_cat)
        parameter = None
        for t in parameters:
            if self._model_selection.value in t:
                parameter = t
                break

        if parameter is None:
            self._param_sel.options = []
            return
        param = self._ml_repo.get(parameter)
        self._param_sel.options = [x for x in param.get_params().keys()]

    def __init__(self, ml_repo):
        self._ml_repo = widget_repo.ml_repo
        self._output = widgets.Output(
            layout=Layout(width='100%', height='100%'))
        self._model_selection = widgets.RadioButtons(
            options=self._ml_repo.get_names(MLObjectType.MODEL),
            #     value='pineapple',
            description='model:',
            disabled=False
        )

        self._train_model_sel = widgets.RadioButtons(
            options=['training', 'model'],
            value='model',
            description='param type:',
            disabled=False
        )
        self._param_sel = widgets.Select(
            description='parameter',
            disabled=False
        )
        self._get_parameter()

        self._train_model_sel.observe(self._get_parameter, 'value')
        self._data_selection = widgets.SelectMultiple(options=[])
        self._get_measures()

        self._plot_button = widgets.Button(description='plot')
        self._plot_button.on_click(self.plot)
        self._widget_main = widgets.VBox(children=[
            widgets.HBox(
                children=[self._plot_button, self._data_selection,
                          self._model_selection, self._train_model_sel, self._param_sel]
            ),
            self._output]
        )

    def get_widget(self):
        return self._widget_main

    def plot(self, d):
        with self._output:
            clear_output(wait=True)
            data = [x for x in self._data_selection.value]
            if len(data) > 0 and self._param_sel.value is not None:
                use_train_param = True
                if self._train_model_sel.value == 'model':
                    use_train_param = False
                plot.measure_by_parameter(self._ml_repo, data, self._param_sel.value,
                                          data_versions=LAST_VERSION, training_param=use_train_param)

