from ipywidgets import *
import ipywidgets as widgets
import numpy as np
import copy
from IPython.display import display, clear_output

from pailab import MLObjectType, RepoInfoKey, FIRST_VERSION, LAST_VERSION
import pailab.tools.checker as checker
import pailab.tools.tools as tools
import pandas as pd

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
    
    def __init__(self):
        pass

    def set_repo(self, ml_repo):
        self.ml_repo = ml_repo
        self._setup()

    def _setup(self):
        self.object_types = {}
        for k in MLObjectType:
            self.object_types[k.value] = self.ml_repo.get_names(k)
        self._setup_data()
        self._setup_measures()

    def _setup_labels(self):
        pass

    def _setup_data(self):
        self.data = [k for k in self.ml_repo.get_names(MLObjectType.TRAINING_DATA)]
        self.data.extend([k for k in self.ml_repo.get_names(MLObjectType.TEST_DATA)])

    def _setup_measures(self):
        measure_names =   self.ml_repo.get_names(MLObjectType.MEASURE_CONFIGURATION)
        if len(measure_names) == 0:
            self.measures = []
        else:
            measure_config = self.ml_repo.get(measure_names[0])
            self.measures = [x for x in measure_config.measures.keys()]

    def _setup_model_info_table(self):
        model_rows = []
        model_names = self.ml_repo.get_names(MLObjectType.CALIBRATED_MODEL)
        for model_name in model_names:
            models = self.ml_repo.get(model_name, version=(FIRST_VERSION, LAST_VERSION), full_object=False)
            for model in models:
                tmp = copy.deepcopy(model.repo_info.get_dictionary())
                tmp['model'] = tmp['name']
                del tmp['big_objects']
                del tmp['modifiers']
                del tmp['modification_info']
                #del tmp['category']
                #del tmp['classname']
                model_rows.append(tmp)
        result = pd.DataFrame(model_rows)
        result.set_index(['model', 'version'], inplace=True)
        return result

    def _setup_error_measure_table(self, data_sets, measures):
        tmp = []
        for measure in measures:
            for data in data_sets:
                tmp.append(pd.DataFrame(tools.get_model_measure_list(self.ml_repo,  measure, data)))
                tmp[-1].set_index(['model', 'version'], inplace=True)
        result = self._setup_model_info_table()
        tmp.insert(0,result)
        return pd.concat(tmp, axis=1)

widget_repo = _MLRepoModel()

def _highlight_max(data, color='red'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'color: {}'.format(color)
    #remove % and cast to float
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
    #remove % and cast to float
    #data = data.replace('%','', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.min()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.min().min()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

class _TableViewer:
    def __init__(self, table, table_name, selected_columns = None):
        self._table = table
        self._table_name = table_name
        self._columns = table.columns
        if selected_columns is None:
            self._selected_columns = self._columns
        else:
            self._selected_columns = selected_columns
        
        self._selected_columns = widgets.SelectMultiple(options = self._columns, value = self._selected_columns)
        self._output=widgets.Output()

        self._settings = widgets.HBox(children = [])
        self._tab = widgets.Tab(children = [self._output, self._settings], title=['Table', 'Table Settings'])

        self._button_update=widgets.Button(description = 'update')
        self._button_update.on_click(self.get_overview)

    def get_overview(self, d):
        with self._output:
            clear_output(wait = True)
            TableDisplay(self._table[self._selected_columns.value])  # , orient='index'))

    def get_widget(self):
        return self._tab

class _ObjectCategorySelector:

    def __init__(self, *args, **kwargs):
        selection = []
        for k,v in widget_repo.object_types.items():
            if len(v) > 0:
                selection.append(k + ' (' + str(len(v)) + ')' )
        if 'layout' not in kwargs.keys():
            kwargs['layout'] = widgets.Layout(width='300px', height = '250px')
        kwargs['value'] = []
        self._selector = widgets.SelectMultiple(options=selection, **kwargs)

    def get_selection(self):
        return [k.split(' ')[0] for k in self._selector.value]

    def get_widget(self):
        return widgets.VBox(children=
                        [
                            widgets.Label(value='Object Types'),
                            self._selector
                        ]
                    ) 

class _DataSelector:
    """Widget to select training and test data.
    """
    def __init__(self, **kwargs):
        self._selection_widget = widgets.SelectMultiple(options = widget_repo.data, **kwargs)
            
    def get_widget(self):
        return widgets.VBox(children=[widgets.Label(value='Data'), self._selection_widget])

    def get_selection(self):
        return self._selection_widget.value


class _MeasureSelector:
    """Widget to select training and test data.
    """
    def __init__(self, **kwargs):
        self._selection_widget = widgets.SelectMultiple(options = widget_repo.measures, **kwargs)
            
    def get_widget(self):
        return widgets.VBox(children=[widgets.Label(value='Measures'), self._selection_widget])

    def get_selection(self):
        return self._selection_widget.value

class ObjectOverviewList:
    def __init__(self, beakerX=False):
        self._categories = _ObjectCategorySelector(layout = widgets.Layout(width='250px', height = '250px'))
        self._repo_info=widgets.SelectMultiple(
            options = [k.value for k in RepoInfoKey], value = ['category', 'name', 'commit_date', 'version'], layout = widgets.Layout(width='200px',height = '250px')
        )
        #self._settings = widgets.HBox(children=[self.categories, self._repo_info])
        
        self._button_update=widgets.Button(description = 'update')
        self._button_update.on_click(self.get_overview)
        
       
        self._output=widgets.Output()
        self._input_box=widgets.HBox(
            children = [
                self._categories.get_widget(), 
                widgets.VBox(children=
                        [
                            widgets.Label(value='Info Fields'),        
                            self._repo_info
                        ]
                ),
                widgets.VBox(children=
                        [
                            self._button_update,
                            self._output
                        ]
                )
            ]
        )

    def get_overview(self, d):
        result={}
        for info in self._repo_info.value:
            result[info]=[]

        for k in self._categories.get_selection():
            for n in  widget_repo.object_types[k]:
                obj= widget_repo.ml_repo.get(n)
                for info in self._repo_info.value:
                    if isinstance(obj.repo_info[info], MLObjectType):
                        result[info].append(obj.repo_info[info].value)
                    else:
                        result[info].append(str(obj.repo_info[info]))
        with self._output:
            clear_output(wait = True)
            TableDisplay(pd.DataFrame.from_dict(result))  # , orient='index'))
            
    def get_widget(self):
        return self._input_box

class ObjectView:

    def _setup_names(self, change = None):
        names=[]
        for k in self._categories.get_selection():
            names.extend(widget_repo.ml_repo.get_names(k))
        self._names.options=names

    def __init__(self):
        self._categories = _ObjectCategorySelector()
        self._names=widgets.SelectMultiple(
            options = []
        )
        self._setup_names()
        self._categories.observe(self._setup_names, 'value')

        self._button_update=widgets.Button(description = 'show history')
        self._button_update.on_click(self.show_history)
        self._output=widgets.Output()
        self._input_box=widgets.HBox(
            children = [self._categories.get_widget(), self._names, self._button_update, self._output]
        )

    def show_history(self, d):
        result={RepoInfoKey.NAME.value: [],
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

    def get_widget(self):
        return self._input_box

class MeasureView:
    def __init__(self, beakerX=False):
        self._data = _DataSelector()
        self._measures = _MeasureSelector()
        self._repo_info=widgets.SelectMultiple(
            options = [k.value for k in RepoInfoKey], value = ['category', 'name', 'commit_date', 'version'], layout = widgets.Layout(width='200px',height = '250px')
        )
        self._output = widgets.Output(layout = widgets.Layout(width='1000px',height = '450px', overflow_y='auto', overflow_x='auto'))
        self._button_update = widgets.Button(description = 'update')    
        self._button_update.on_click(self.get_measures)
    
    def _get_columns_selected(self):
        columns = [x for x in self._repo_info.value]
        for data in self._data.get_selection():
            for m in self._measures.get_selection():
                columns.append(m+', '+data)
        return columns

    def get_widget(self):
        self._tab = widgets.Tab(children=[
                        self._output,
                        widgets.HBox(children = 
                        [
                            self._data.get_widget(), 
                            self._measures.get_widget(), 
                            widgets.VBox(children=[
                                widgets.Label(value='Model Columns'),
                                self._repo_info]
                            ),
                            self._button_update
                        ])                               
                    ],
                    title=['Table', 'Settings']
                    )
        self._tab.set_title(0,'Table')
        self._tab.set_title(1,'Settings')
        return self._tab

        return widgets.HBox( children = [ 
                widgets.VBox(children = 
                    [
                        self._data.get_widget(), 
                        self._measures.get_widget(), 
                        self._repo_info,
                        self._button_update
                    ]),
                self._output
            ]
        )

    def get_measures(self, d):
        self._tab.selected_index = 0
        tmp = widget_repo._setup_error_measure_table(self._data.get_selection(), self._measures.get_selection())
        columns = [c for c in tmp.columns if c in self._get_columns_selected() ]
        tmp2 = tmp[columns]
        with self._output:
            clear_output(wait = True)
            # apply highlighting to floating columns only
            floats = [x.kind == 'f' for x in tmp2.dtypes]
            float_columns = tmp2.columns[floats]
            TableDisplay(tmp2.style.apply(_highlight_max, subset =float_columns).apply(_highlight_min, subset = float_columns))  # , orient='index'))
    
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
            print('test issues: ' + str(len(self._test_results)) + ', model issues: ' + str(len(self._model_results))+ ', data issues: ' + str(len(self._data_results)))
            result = {'test': [], 'message':[], 'model': [], 'type': []}
            for k,v in self._test_results.items():    
                for a,b in v.items():
                    result['type'].append('test')
                    result['model'].append(k)
                    result['test'].append(a)
                    result['message'].append(b)
            for k,v in self._model_results.items():    
                for a,b in v.items():
                    result['type'].append('model')
                    result['model'].append(k)
                    result['test'].append(a)
                    result['message'].append(b)
            for k,v in self._data_results.items():    
                for a,b in v.items():
                    result['type'].append('data')
                    result['model'].append(k)
                    result['test'].append(a)
                    result['message'].append(b)
           
            display(pd.DataFrame.from_dict(result))
    def get_widget(self):
        return self._widget_main


import pailab.analysis.plot_helper as plt_helper
import pailab.analysis.plot as plot

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
        if self._train_model_sel.value=='model':
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
        self._output = widgets.Output(layout=Layout(width='100%', height='100%'))
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
        self._widget_main = widgets.VBox(  children=[ 
                widgets.HBox(
                    children=[self._plot_button,self._data_selection, self._model_selection, self._train_model_sel, self._param_sel]
                        ), 
                self._output] 
            )
        

    def get_widget(self):
        return self._widget_main

    def plot(self, d):
        with self._output:
            clear_output(wait=True)
            data = [ x for x in self._data_selection.value]
            if len(data) > 0 and self._param_sel.value is not None:
                use_train_param = True
                if self._train_model_sel.value == 'model':
                    use_train_param = False
                plot.measure_by_parameter(self._ml_repo, data, self._param_sel.value, data_versions= LAST_VERSION, training_param = use_train_param)

