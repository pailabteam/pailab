from ipywidgets import *
import ipywidgets as widgets
from IPython.display import display, clear_output

from pailab import MLObjectType, RepoInfoKey, FIRST_VERSION, LAST_VERSION
import pailab.checker as checker
import pandas as pd

pd.set_option('display.max_colwidth', -1) #set option so that long lines have a linebreak

beakerX = False
if beakerX:
    from beakerx import *
    from beakerx.object import beakerx
else:
    def TableDisplay(dt):
        display(dt)


class ObjectOverviewList:
    def __init__(self, ml_repo, beakerX=False):
        self._ml_repo = ml_repo
        self._categories = widgets.SelectMultiple(
            options=[k.value for k in MLObjectType])
        self._repo_info = widgets.SelectMultiple(
            options=[k.value for k in RepoInfoKey], value=['category', 'name']
        )
        self._button_update = widgets.Button(description='update')
        self._button_update.on_click(self.get_overview)
        self._output = widgets.Output()
        self._input_box = widgets.HBox(
            children=[widgets.VBox(
                children=[self._categories, self._repo_info]), self._button_update, self._output]
        )

        #

        # self.select_category = widgets.RadioButtons(
        #     options=['ALL', 'Model', 'Result', 'Job', 'RawData', 'DataSet', 'TrainingParameter', 'Test', 'Pipeline', 'ModelParameter'])
        # self.cb_additional_fields = []
        # for i in ['status', 'modification_info']:
        #     self.cb_additional_fields.append(widgets.Checkbox(
        #         description=i, value=False, width=1300))
        # self.objects = widgets.VBox(children=self.cb_additional_fields)
        # self.selection_overall = widgets.HBox(
        #     children=[self.select_category, self.objects])
        # self.button_show_objects = widgets.Button(description='update')
        # self.selection = widgets.VBox(
        #     children=[self.selection_overall, self.button_show_objects])

        # self.out_objects_stored = widgets.Output()
        # self.out_objects_overview = widgets.Tab(
        #     children=[self.selection, self.out_objects_stored])
        # self.out_objects_overview.set_title(0, 'settings')
        # self.out_objects_overview.set_title(1, 'overview')
        # self.final_widget = self.out_objects_overview
        # self.button_show_objects.on_click(self.get_overview)

    def get_overview(self, d):
        result = {}
        for info in self._repo_info.value:
            result[info] = []

        for k in self._categories.value:
            names = self._ml_repo.get_names(k)
            for n in names:
                obj = self._ml_repo.get(n)
                for info in self._repo_info.value:
                    if isinstance(obj.repo_info[info], MLObjectType):
                        result[info].append(obj.repo_info[info].value)
                    else:
                        result[info].append(str(obj.repo_info[info]))
        with self._output:
            clear_output(wait=True)
            TableDisplay(pd.DataFrame.from_dict(result))  # , orient='index'))

        # _object_type = self.select_category.value
        # if _object_type == 'ALL':
        #     _object_type = ''
        # fields = []
        # for x in self.cb_additional_fields:
        #     if x.value:
        #         fields.append(x.description)
        # overview = repo.get_overview(_object_type, fields)
        # for version, x in overview.items():
        #     for key, value in x.items():
        #         if isinstance(value, dict):
        #             x[key] = str(value)
        # with self.out_objects_stored:
        #     clear_output(wait=True)
        #     display(pd.DataFrame.from_dict(overview, orient='index'))
        # self.out_objects_overview.selected_index = 1
        # # return pd.DataFrame.from_dict(overview,orient='index')
    # display()

    def get_widget(self):
        return self._input_box
        # self.cb_additional_fields = []
        # return self.final_widget

class ObjectView:

    def _setup_names(self, change=None):
        names = []
        for k in self._categories.value:
            names.extend(self._ml_repo.get_names(k))
        self._names.options = names

    def __init__(self, ml_repo, beakerX=False):
        self._ml_repo = ml_repo
        self._categories = widgets.SelectMultiple(
            options=[k.value for k in MLObjectType], value=['CALIBRATED_MODEL'])
        self._names = widgets.SelectMultiple(
            options=[]
        )
        self._setup_names()
        self._categories.observe(self._setup_names, 'value')

        self._button_update = widgets.Button(description='show history')
        self._button_update.on_click(self.show_history)
        self._output = widgets.Output()
        self._input_box = widgets.HBox(
            children=[widgets.VBox(
                children=[self._categories, self._names]), self._button_update, self._output]
        )

    def show_history(self, d):
        result={  RepoInfoKey.NAME.value : [],
             RepoInfoKey.AUTHOR.value : [], 
             RepoInfoKey.VERSION.value : [], 
             RepoInfoKey.COMMIT_DATE.value : []}
        for k in self._names.value:
            history = self._ml_repo.get_history(k)
            for l in history:
                for m in result.keys():
                    result[m].append(l['repo_info'][m])
        with self._output:
            clear_output(wait=True)
            TableDisplay(pd.DataFrame.from_dict(result))

    def get_widget(self):
        return self._input_box

class ConsistencyChecker:

    def _consistency_check(self):
        self._test_results = checker.Tests.run(self._ml_repo)
        self._model_results = checker.Model.run(self._ml_repo)
        self._data_results =checker.Data.run(self._ml_repo)

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


import pailab.plot_helper as plt_helper
import pailab.plot as plot
#import plotly.graph_objs as go
#from plotly.offline import download_plotlyjs, init_notebook_mode, iplot


class Plotter:
    def _get_measures(self):
        return self._ml_repo.get_names(MLObjectType.MEASURE)

    def __init__(self, ml_repo):
        self._ml_repo = ml_repo
        self._output = widgets.Output(layout=Layout(width='100%', height='100%'))
        
        self._data_selection = widgets.SelectMultiple(options=self._get_measures()) 
        self._plot_button = widgets.Button(description='plot')
        self._plot_button.on_click(self.plot)
        self._widget_main = widgets.VBox(  children=[ widgets.HBox(children=[self._plot_button,self._data_selection]), self._output] )
        

    def get_widget(self):
        return self._widget_main

    def plot(self, d):
        with self._output:
            clear_output(wait=True)
        
            plot.measure_by_parameter(self._ml_repo,[ x for x in self._data_selection.value], 'max_depth', data_versions= LAST_VERSION)

