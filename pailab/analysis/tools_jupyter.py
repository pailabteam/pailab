import numpy as np
import copy
import logging
from IPython.display import display, clear_output
from collections import defaultdict
import pailab.analysis.plot as paiplot
import pailab.analysis.plot_helper as plt_helper
import ipywidgets as widgets

from pailab import MLObjectType, RepoInfoKey, FIRST_VERSION, LAST_VERSION
from pailab.ml_repo.repo import NamingConventions
import pailab.tools.checker as checker
import pailab.tools.tools as tools
import pailab.tools.interpretation as interpretation
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

# set option so that long lines have a linebreak
pd.set_option('display.max_colwidth', -1)
# set widget use to True so that plotlys FigureWidget is used
paiplot.use_within_widget = True

if paiplot.has_plotly:
    import plotly.graph_objs as go

beakerX = False
if beakerX:
    from beakerx import TableDisplay
    # from beakerx.object import beakerx
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
            self.labels = {}  # dictionary label->model and version
            # dictionary (model,version)->labelname or None
            self.model_to_label = defaultdict(lambda: None)
            self._setup_labels(ml_repo)
            self._model_info_table = self._setup_model_info_table(ml_repo)
            self._model_names = ml_repo.get_names(
                MLObjectType.CALIBRATED_MODEL)

        def _setup_labels(self, ml_repo):
            label_names = ml_repo.get_names(MLObjectType.LABEL)
            if label_names is None:
                return
            if isinstance(label_names, str):
                label_names = [label_names]
            for l in label_names:
                label = ml_repo.get(l)
                self.labels[l] = {'model': label.name,
                                  'version': label.version}
                self.model_to_label[(label.name, label.version,)] = l

        def _setup_model_info_table(self, ml_repo):
            model_rows = []
            model_names = ml_repo.get_names(MLObjectType.CALIBRATED_MODEL)
            for model_name in model_names:
                models = ml_repo.get(model_name, version=(
                    FIRST_VERSION, LAST_VERSION), full_object=False)
                if not isinstance(models, list):
                    models = [models]
                for model in models:
                    tmp = copy.deepcopy(model.repo_info.get_dictionary())
                    tmp['model'] = tmp['name']
                    del tmp['big_objects']
                    del tmp['modifiers']
                    del tmp['modification_info']
                    tmp['label'] = self.model_to_label[(
                        tmp['model'], tmp['version'],)]
                    tmp['widget_key'] = tmp['commit_date'][0:16] + ' | ' + \
                        tmp['author'] + ' | ' + \
                        str(tmp['label']) + ' | ' + tmp['version']
                    model_rows.append(tmp)
            model_info_table = pd.DataFrame(model_rows)
            model_info_table.set_index(['model', 'version'], inplace=True)
            return model_info_table

        def get_models(self):
            return self._model_names

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
        # now set label information into

    def _setup_labels(self):  # todo: das hier muss weg
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

    def get_versions(self, name):
        return self.ml_repo.get_history(name, obj_member_fields=[])

    # def get_model_parameter(self, model_name)


widget_repo = _MLRepoModel()

# region helpers


def _add_title_and_border(name):
    def _get_widget(get_widget):
        def wrapper(self):
            return widgets.VBox(children=[
                # , layout = widgets.Layout(width = '100%')),
                widgets.HTML(
                    value='<h3 style="Color: white; background-color:#d1d1e0; text-align: center"> ' + name + '</h3>'),
                get_widget(self),
                # , layout = widgets.Layout(width = '100%'))
                widgets.HTML(
                    value='<h3 style="Color: white; background-color:#d1d1e0; text-align: center"> </h3>')
            ], layout=widgets.Layout(padding='0px 0px 0px 0px', overflow_x='auto')  # , overflow_y='auto', )
            )  # layout=widgets.Layout(border='solid 1px'))
        return wrapper
    return _get_widget


def _highlight_max(data, color='red'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'color: {}'.format(color)
    # remove % and cast to float
    # data = data.replace('%','', regex=True).astype(float)
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
    # data = data.replace('%','', regex=True).astype(float)
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
        self._selector = widgets.SelectMultiple(options=selection,
                                                # value = [selection[0]],
                                                **kwargs)

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
        names = widget_repo.data.get_data_names()
        # if len(names) > 0:
        self._selection_widget = widgets.SelectMultiple(
            options=names, value=[names[0]], **kwargs)

    def get_widget(self):
        return widgets.VBox(children=[widgets.Label(value='Data'), self._selection_widget])

    def get_selection(self):
        return self._selection_widget.value


class _DataSelectorWithVersion:
    """Widget to select training and test data.
    """

    def __init__(self, display_selection=True, **kwargs):
        names = widget_repo.data.get_data_names()
        self._update_callbacks = []
        self._display_selection = display_selection
        self._selection = {}
        self._selection_options = {}
        self._key_to_version = {}
        self._updating_version = {}
        for n in names:
            self._selection[n] = []
            self._selection_options[n] = []
            self._key_to_version[n] = {}
        self._selected_overview = widgets.Output()
        self._selection_data = widgets.Dropdown(
            options=names, value=None, **kwargs)

        self._selection_data.observe(self._update_version, names='value')

        self._selection_version = widgets.SelectMultiple(
            options=[], value=[], **kwargs)
        self._selection_version.observe(
            self._display_selected_overview, names='value')

    def _get_state(self):
        return self._selection, self._selection_options,  self._key_to_version

    def _set_state(self, state):
        self._selection = state[0]
        self._selection_options = state[1]
        self._key_to_version = state[2]

    def _set_update_callback(self, cb):
        """Set a callback (called at every update of this widget)

        Args:
            cb (function): Callback function called at every update.
        """
        self._update_callbacks.append(cb)

    def _update_version(self, change):
        self._updating_version = True
        data_selected = self._selection_data.value
        tmp = widget_repo.ml_repo.get_history(data_selected)
        key_to_version = {}
        versions = []
        for x in tmp:
            key = x['repo_info']['commit_date'][0:16] + ' | ' + \
                x['repo_info']['author'] + ' | ' + x['repo_info']['version']
            key_to_version[key] = x['repo_info']['version']
            versions.append(key)
        self._key_to_version[data_selected] = key_to_version
        self._selection_version.options = versions
        self._selection_version.value = self._selection_options[data_selected]
        for cb in self._update_callbacks:
            cb(change)
        self._updating_version = False
        # self._selection[self._selection_data.value] = [x for x in self._selection_version.value]

    def _display_selected_overview(self, change):
        if self._updating_version:
            return
        data_selected = self._selection_data.value
        key_to_version = self._key_to_version[data_selected]
        self._selection[data_selected] = [key_to_version[x]
                                          for x in self._selection_version.value]
        self._selection_options[data_selected] = [
            x for x in self._selection_version.value]
        tmp = {}
        tmp['data'] = []
        tmp['version'] = []
        for n, x in self._selection.items():
            for y in x:
                tmp['data'].append(n)
                tmp['version'].append(y)
        for cb in self._update_callbacks:
            cb(change)
        with self._selected_overview:
            clear_output(wait=True)
            display(pd.DataFrame.from_dict(tmp))

    def get_widget(self):
        if self._display_selection:
            return widgets.VBox(children=[widgets.Label(value='Data'), self._selection_data,
                                          widgets.Label(
                                              value='Versions'), self._selection_version,
                                          self._selected_overview, ])
        else:
            return widgets.VBox(children=[widgets.Label(value='Data'), self._selection_data,
                                          widgets.Label(value='Versions'), self._selection_version])

    def get_selection(self):
        return self._selection

    def get_data(self):
        data = {}
        for d_name, d_v in self._selection.items():
            if len(d_v) > 0:
                data[d_name] = d_v
        return data


class _ModelSelectorWithVersion:

    @staticmethod
    def _filter_models(labels=None, commit_start=None, commit_end=None, authors=None, model_versions=None):
        """Filter the model table according to the given attributes.

        Args:
            labels ([str or iterable of str], optional): If set, returns only models with the selected labels. Defaults to None.
            commit_start (str, optional): String of earliest commit date.. Defaults to None.
            commit_end (str, optional): String of latest commit date. Defaults to None.
            authors (str or iterable of str, optional): If set it return only the models with the corresponding author(s). Defaults to None.
            model_versions (str or iterable of str, optional): If set only modes with respective version(s) are returned. Defaults to None.

        Returns:
            pandas DataFrame: The correspondign models.
        """
        result = widget_repo.model.get_info_table()
        if labels is not None:
            if isinstance(labels, str):
                result = result[result['label'] == labels]
            else:
                result = result[result['label'].isin(labels)]
        if commit_start is not None:
            result = result[result['commit_date'] >= commit_start]
        if commit_end is not None:
            result = result[result['commit_date'] <= commit_end]
        if authors is not None:
            if isinstance(authors, str):
                result = result[result['author'] == authors]
            else:
                result = result[result['author'].isin(authors)]
        if model_versions is not None:
            if isinstance(model_versions, str):
                result = result[result['version'] == model_versions]
            else:
                result = result[result['version'].isin(model_versions)]
        return result

    def __init__(self,  display_selection=True, **kwargs):
        self._display_selection = display_selection
        self._selection = defaultdict(list)
        self._selection_model_name = widgets.Dropdown(
            options=widget_repo.model.get_models(), value=None, **kwargs)
        self._selection_model_name.observe(
            self._selected_model_changes, names='value')

        self._selection_version = widgets.SelectMultiple(
            options=[], value=[], rows=8, layout=widgets.Layout(width="100%"), **kwargs)

        self._selected_overview = widgets.Output()
        self._selection_version.observe(
            self._selected_version_changed, names='value')

        self._model_changed_callable = None

        # Filtering
        #
        labels = widget_repo.ml_repo.get_names(MLObjectType.LABEL)
        self._label_selector = widgets.SelectMultiple(options=labels)
        self._commit_data_start = widgets.DatePicker()
        self._commit_data_end = widgets.DatePicker()
        self._author_selector = widgets.SelectMultiple(
            options=widget_repo.model.get_info_table()['author'].unique())
        self._apply_button = widgets.Button(description='Apply')
        self._apply_button.on_click(self._apply_filter)
        self._clear_button = widgets.Button(description='Clear')
        self._clear_button.on_click(self._clear_filter)
        self._filter = widgets.VBox(children=[
            widgets.Label(value='Labels'),
            self._label_selector,
            widgets.Label(value='Commit Start'),
            self._commit_data_start,
            widgets.Label(value='Commit End'),
            self._commit_data_end,
            widgets.Label(value='Authors'),
            self._author_selector,
            widgets.HBox(children=[
                self._apply_button,
                self._clear_button])
        ]
        )

    def get_models(self):
        """Returns all selected models as list of tuples (first element is model name, second model version)
        """
        models = widget_repo.model.get_info_table()
        result = {}
        for k, v in self._selection.items():
            if len(v) > 0:
                result[k] = [models[models['widget_key'] == w].index[0][1]
                             for w in v]
        return result

    def observe_model_change(self, handler):
        """Setup a handler when the model trait changed

        Args:
            handler (callable): A callable that is called when the model trait changes.
        """
        self._model_changed_callable = handler

    def _selected_model_changes(self, change):
        self._update_version(change)
        if self._model_changed_callable is not None:
            self._model_changed_callable(change)

    def _selected_version_changed(self, change):
        self._display_selected_overview(change)

    def _apply_filter(self, dummy):
        self._updating_version = True
        data_selected = self._selection_model_name.value
        labels = self._label_selector.value
        if len(labels) == 0:
            labels = None
        if self._commit_data_start.value is None:
            commit_start = None
        else:
            commit_start = str(self._commit_data_start.value)
        if self._commit_data_end.value is None:
            commit_end = None
        else:
            commit_end = str(self._commit_data_end.value)
        authors = None
        if len(self._author_selector.value) > 0:
            authors = self._author_selector.value
        models = _ModelSelectorWithVersion._filter_models(labels=labels, authors=authors,
                                                          commit_start=commit_start, commit_end=commit_end)
        self._selection_model_name.options = [
            x for x in models['name'].unique()]
        models = models[models['name'] == data_selected]
        widget_keys = models['widget_key'].values
        self._selection_version.options = [x for x in models['widget_key']]
        self._selection_version.value = [
            x for x in self._selection[data_selected] if x in widget_keys]
        self._updating_version = False

    def _clear_filter(self, dummy):
        self._commit_data_start.value = None
        self._commit_data_end.value = None
        self._author_selector.value = []
        self._label_selector.value = []
        self._apply_filter(dummy)

    def _update_version(self, change):
        if change['old'] is not None:
            pass
        self._updating_version = True
        data_selected = self._selection_model_name.value
        models = widget_repo.model.get_info_table()
        models = models[models['name'] == data_selected]
        self._selection_version.options = [x for x in models['widget_key']]
        self._selection_version.value = self._selection[data_selected]
        self._updating_version = False

    def _update_selected_versions(self, change):
        data_selected = self._selection_model_name.value
        # now handle changes of version selection: Remove versions that have been
        # deselected and add versions that have been selected
        old = set(change['old'])
        new = set(change['new'])
        # remove versions that have been deselected
        diff = old-new
        self._selection[data_selected] = list(
            set(self._selection[data_selected])-diff)
        # add new elements
        diff = new - old
        self._selection[data_selected].extend(diff)

    def _display_selected_overview(self, change):
        if self._updating_version:
            return
        self._update_selected_versions(change)
        versions = []
        for n, x in self._selection.items():
            versions.extend(x)
        with self._selected_overview:
            clear_output(wait=True)
            models = widget_repo.model.get_info_table()
            display(models[models['widget_key'].isin(versions)])

    def get_widget(self):
        filter_widget = widgets.Accordion(
            children=[self._filter], selected_index=None)
        filter_widget.set_title(0, 'Filter')
        if self._display_selection:
            return widgets.VBox(children=[
                widgets.VBox(children=[
                    widgets.Label(value='Model'),
                    self._selection_model_name,
                    widgets.Label(value='Versions'),
                    self._selection_version,
                    self._selected_overview,
                ]
                ),
                filter_widget])

        else:
            return widgets.VBox(children=[
                widgets.VBox(children=[
                                widgets.Label(value='Model'),
                                self._selection_model_name,
                                widgets.Label(value='Versions'),
                                self._selection_version
                                ]
                             ),
                filter_widget])


class _ModelAndDataSelectorWithVersion:
    """Widget to select a model together with data used in conjunction with the selected model.

    Returns:
        [type]: [description]
    """

    def __init__(self,  display_selection=True, **kwargs):
        self._display_selection = display_selection
        names = widget_repo.model.get_models()
        self._data = _DataSelectorWithVersion(display_selection=False)
        self._model = _ModelSelectorWithVersion(display_selection=False)
        self._data._set_update_callback(self._display_selected_overview)
        self._selected_overview = widgets.Output()

    def get_models(self):
        """Returns all selected models as dictionary from model to list of selected model's versions
        """
        return self._model.get_models()

    def get_data(self):
        return self._data.get_data()

    def _display_selected_overview(self, change):
        # if self._updating_version:
        #    return
        # data_selected = self._selection_data.value
        # key_to_version = self._key_to_version[data_selected]
        # self._selection[data_selected] = [key_to_version[x] for x in self._selection_version.value]
        # self._selection_options[data_selected] = [x for x in self._selection_version.value]
        # tmp ={}
        # tmp['model'] = []
        # tmp['model version'] =[]
        # tmp['data'] = []
        # tmp['data version'] =[]
        # for n, x in self._selection.items():
        #     for y in x:
        #         for data_name, data_versions in self._model_to_data_states[n][0].items():
        #             for data_version in data_versions:
        #                 tmp['model'].append(n)
        #                 tmp['model version'].append(y)
        #                 tmp['data'].append(data_name)
        #                 tmp['data version'].append(data_version)

        # with self._selected_overview:
        #     clear_output(wait = True)
        #     df = pd.DataFrame.from_dict(tmp)
        #     df = df[['model', 'model version', 'data', 'data version']]
        #     #arrays=[tmp['model'],tmp['model version'], tmp['data']]
        #     #df = pd.DataFrame([tmp['data version']], index=arrays)
        #     #multi_index = pd.MultiIndex.from_arrays(arrays, names=('model','model version', 'data', 'data version'))
        #     #df.reindex(index = multi_index)
        #     display(df)
        pass

    def get_widget(self):
        model_selection = widgets.Accordion(
            children=[self._model.get_widget()])
        model_selection.set_title(0, 'Model')
        model_selection.selected_index = None
        data_selection = widgets.Accordion(children=[self._data.get_widget()])
        data_selection.set_title(0, 'Data')
        data_selection.selected_index = None
        if self._display_selection:
            return widgets.VBox(children=[
                model_selection,
                data_selection,
                self._selected_overview, ])
        else:
            return widgets.VBox(children=[
                model_selection,
                data_selection])


class _MeasureSelector:
    """Widget to select measures.
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
            layout=widgets.Layout(width='200px', height='250px', margin='10px')
        )
        # self._settings = widgets.HBox(children=[self.categories, self._repo_info])

        self._button_update = widgets.Button(description='update')
        self._button_update.on_click(self.get_overview)

        self._output = widgets.Output(layout=widgets.Layout(
            height='300px', width='1000px', overflow_y='auto',  overflow_x='auto'))
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
                    layout=widgets.Layout(margin='10px 10px 10px 10px')
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
            + widget_repo.ml_repo._config['name'] + '</h4>')  # , margin = '0px 0px 0px 0px'))
        self._data_statistics = widgets.Output(
            layout=widgets.Layout(width='450px', height='450px'))
        self._plot_data_statistics()
        self._measures = widgets.Output(
            layout=widgets.Layout(width='450px', height='450px'))
        self._consistency = self._setup_consistency()
        self._labels = self._setup_labels()
        self._model_stats = self._setup_model_stats()

    # check consistency

    def _setup_consistency(self):
        def create_consistency_html(**kwargs):
            result = '<div style="background-color:#c2c2d6">'
            result += '<h4 stype="text-align: center">Consistency</h4>'
            for k, v in kwargs.items():
                if len(v) > 0:
                    result += '<p style="background-color:red"> ' + \
                        str(v) + ' ' + k + ' issues found!</p>'
                else:
                    result += '<p style="background-color:lightgreen">No ' + k + ' issues found.</p>'
            result += '</div>'
            return result

        return widgets.HTML(create_consistency_html(model=widget_repo.consistency.model,
                                                    test=widget_repo.consistency.tests,
                                                    data=widget_repo.consistency.data),
                            layout=widgets.Layout(margin='0% 0% 0% 0%', width='400px'))

    def _setup_labels(self):
        header = widgets.HTML(
            '<div style="background-color:#c2c2d6"><h4 stype="text-align: center">Labels</h4>')
        label_output = None

        if len(widget_repo.labels) > 0:
            label_output = widgets.Output(
                layout=widgets.Layout(width='400px', height='100px', overflow_y='auto', overflow_x='auto'))
            with label_output:
                clear_output(wait=True)
                display(pd.DataFrame.from_dict(
                    widget_repo.labels, orient='index'))
        else:
            label_output = widgets.HTML(
                '<div style="background-color:#ff4d4d"><h4 stype="text-align: center">No labels defined.</h4>')

        return widgets.VBox(children=[header, label_output])

    def _setup_model_stats(self):
        header = widgets.HTML(
            '<div style="background-color:#c2c2d6"><h4 stype="text-align: center">Models</h4>')
        model_stats_output = widgets.Output(
            layout=widgets.Layout(width='400px', height='100px', overflow_y='auto', overflow_x='auto'))
        with model_stats_output:
            clear_output(wait=True)
            display(pd.DataFrame.from_dict(
                widget_repo.get_model_statistics(), orient='index'))
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
            # fig.autofmt_xdate()
            plt.legend()
            ax.set_title('Measures')
            # plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
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


class ModelErrorHistogram:
    """Class to plot histograms of model errors.
      Please make sure that if you use plotly, also the jupyter plotlywidgets are installed via:
        jupyter nbextension install --py --sys-prefix plotlywidget
      otherwise you may encounter problems using this class.
    """

    def __init__(self):
        self._model_data_selector = _ModelAndDataSelectorWithVersion(
            display_selection=False)

        self._update_button = widgets.Button(description='update')
        self._update_button.on_click(self._plot)
        self._output = widgets.Output()
        self._coord = widgets.SelectMultiple(
            options=widget_repo.data._y_coord_names,
            value=[widget_repo.data._y_coord_names[0]],
            disabled=False
        )

    def _plot(self, d):
        with self._output:
            clear_output(wait=True)
            display(go.FigureWidget(paiplot.histogram_model_error(widget_repo.ml_repo, self._model_data_selector.get_models(),
                                                                  self._model_data_selector.get_data(), y_coordinate=self._coord.value)))

    @_add_title_and_border('Pointwise Model Error Histogram')
    def get_widget(self):
        y_coord = widgets.Accordion(children=[self._coord])
        y_coord.set_title(0, 'Y-coordinates')
        return widgets.HBox(children=[
            widgets.VBox(children=[
                self._model_data_selector.get_widget(),
                y_coord,
                self._update_button
            ]),
            self._output
        ])


class ModelErrorConditionalHistogram:
    """Plots the distribution of input data along a given axis for the largest absolute pointwise errors in comparison to the distribution of all data.
    """

    def __init__(self):
        self._data_model_selection = _ModelAndDataSelectorWithVersion(
            display_selection=False)
        self._update_button = widgets.Button(description='update')
        self._update_button.on_click(self._plot)
        self._output = widgets.Output()
        self._recommendation_output = widgets.Output()
        self._recommendation_table = None
        self._output_tab = widgets.Tab(children=[self._output,
                                                 self._recommendation_output])
        self._output_tab.set_title(0, 'histograms')
        self._output_tab.set_title(1, 'recommendations')
        self._quantile = widgets.FloatSlider(
            value=10,
            min=1,
            max=50,
            step=1,
            readout=True,
            readout_format='.2f',
        )
        self._coord = widgets.Select(
            options=widget_repo.data._y_coord_names,
            value=widget_repo.data._y_coord_names[0],
            disabled=False
        )
        self._x_coord = widgets.Select(
            options=widget_repo.data._x_coord_names,
            value=widget_repo.data._x_coord_names[0],
            disabled=False
        )
        self._accordion = widgets.Accordion(children=[
            self._get_selection_widget(),
            self._get_recommendation_widget()
        ])
        self._accordion.set_title(0, 'Selection')
        self._accordion.set_title(1, 'Recommendation')

    def _get_selection_widget(self):
        coordinate_selection = widgets.Accordion(children=[
            widgets.VBox(children=[
                widgets.Label(value='y-coordinates'),
                self._coord,
                widgets.Label(value='x-coordinates'),
                self._x_coord])
        ])
        coordinate_selection.set_title(0, 'Coordinates')
        return widgets.VBox(children=[
            self._data_model_selection.get_widget(),
            coordinate_selection,
            self._quantile,
            self._update_button])

    def _get_recommendation_widget(self):
        self._update_recommendation = widgets.Button(description='update')
        self._max_num_recommendations = widgets.IntText(value=20,
                                                        description='maximum number of recommendations')
        self._cache_in_repo = widgets.Checkbox(
            value=True, description='cache MMD in repo')
        self._scale = widgets.Checkbox(
            value=True, description='scale x-values to zero mean and unit variance')
        self._update_recommendation.on_click(self._recommend)
        self._kernel_selection = widgets.Dropdown(options=[
            'rbf', 'linear', 'polynomial', 'sigmoid', 'laplacian', 'chi2'
        ],
            value='rbf',
            description='kernel')
        self._gamma = widgets.FloatText(value=1.0, description='gamma')
        self._gamma_for_kernel = [
            'rbf', 'polynomial', 'sigmoid', 'laplacian', 'chi2']
        self._kernel_selection.observe(self._on_kernel_change, names='value')
        self._recommendation_selection = widgets.IntText(
            description='recommendation id')
        self._recomendation_selection_apply = widgets.Button(
            description='apply recommendation')
        self._recomendation_selection_apply.on_click(self._apply_recommend)
        return widgets.VBox(children=[
            self._max_num_recommendations,
            self._cache_in_repo,
            self._scale,
            self._kernel_selection,
            self._gamma,
            self._update_recommendation,
            self._recommendation_selection,
            self._recomendation_selection_apply
        ])
        self._recommendation_table = None

    def _on_kernel_change(self, d):
        if self._kernel_selection in self._gamma_for_kernel:
            self._gamma.disabled = False
        else:
            self._gamma.disabled = True

    def _apply_recommend(self, d):
        if self._recommendation_table is None:
            logger.error(
                'Recommendation table is empty, please first update the recommendation.')
            with self._output:
                clear_output(wait=True)
                print(
                    'Recommendation table is empty, please first update the recommendation.')
            return

        if self._recommendation_selection.value is not None:
            self._coord.value = self._recommendation_table['y-coord'][self._recommendation_selection.value]
            self._x_coord.value = self._recommendation_table[
                'x-coord'][self._recommendation_selection.value]
            self._models.value = [
                self._recommendation_table['model'][self._recommendation_selection.value]]
            self._data.value = [
                self._recommendation_table['data'][self._recommendation_selection.value]]
            self._plot(None)

    def _plot(self, d):
        with self._output:
            clear_output(wait=True)
            display(go.FigureWidget(
                    paiplot.histogram_data_conditional_error(widget_repo.ml_repo,
                                                             self._data_model_selection.get_models(), self._data_model_selection.get_data(),
                                                             x_coordinate=self._x_coord.value,
                                                             y_coordinate=self._coord.value,
                                                             percentile=self._quantile.value/100.0)
                    ))
            self._output_tab.selected_index = 0

    def _recommend(self, d):
        self._output_tab.set_title(1, 'computing...')
        self._recommendation_table = pd.DataFrame.from_dict(
            plt_helper.get_ptws_error_dist_mmd(widget_repo.ml_repo, self._data_model_selection.get_models(),
                                               data=self._data_model_selection.get_data(),
                                               start_index=0, end_index=-1, percentile=self._quantile.value/100.0,
                                               scale=self._scale.value,
                                               cache=self._cache_in_repo,
                                               metric=self._kernel_selection.value,
                                               gamma=self._gamma.value)
        )
        self._recommendation_table['model version']
        self._recommendation_table['data version']
        self._recommendation_table.sort_values(
            ['mmd'], ascending=False, inplace=True)
        with self._recommendation_output:
            clear_output(wait=True)
            display(
                self._recommendation_table.iloc[0:self._max_num_recommendations.value])
        self._output_tab.selected_index = 1
        self._output_tab.set_title(1, 'recommendations')
        self._recommendation_selection.value = self._recommendation_table.index[0]

    @_add_title_and_border('Data Distribution of Largest Pointwise Errors.')
    def get_widget(self):
        return widgets.HBox(children=[
            self._accordion,
            self._output_tab
        ])


class ScatterModelError:
    def __init__(self):
        self._model_data_selector = _ModelAndDataSelectorWithVersion(
            display_selection=False)
        self._update_button = widgets.Button(description='update')
        self._update_button.on_click(self._plot)
        self._output = widgets.Output()
        self._coord = widgets.Select(
            options=widget_repo.data._y_coord_names,
            value=widget_repo.data._y_coord_names[0],
            disabled=False
        )
        self._x_coord = widgets.Select(
            options=widget_repo.data._x_coord_names,
            value=widget_repo.data._x_coord_names[0],
            disabled=False
        )

    def _get_selection_widget(self):
        coordinates = widgets.Accordion(children=[
            widgets.VBox(children=[
                widgets.Label(value='y-coordinates'),
                self._coord,
                widgets.Label(value='x-coordinates'),
                self._x_coord,
            ]
            )
        ]
        )
        coordinates.set_title(0, 'Coordinates')
        return widgets.VBox(children=[
            self._model_data_selector.get_widget(),
            coordinates,
            self._update_button]
        )

    def _plot(self, d):
        with self._output:
            clear_output(wait=True)
            display(go.FigureWidget(
                    paiplot.scatter_model_error(widget_repo.ml_repo,
                                                self._model_data_selector.get_models(),
                                                self._model_data_selector.get_data(),
                                                x_coordinate=self._x_coord.value,
                                                y_coordinate=self._coord.value)
                    ))

    @_add_title_and_border('Scatter Plot Pointwise Errors.')
    def get_widget(self):
        return widgets.HBox(children=[
            self._get_selection_widget(),
            self._output
        ])


class IndividualConditionalExpectation:
    """Plots the individual conditional expectation at a certain point.
    """

    def __init__(self):
        names = widget_repo.data.get_data_names()
        self._model_data_selection = _ModelAndDataSelectorWithVersion()
        self._update_button = widgets.Button(description='update')
        self._update_button.on_click(self._plot)
        self._output = widgets.Output()
        self._cluster_statistics_output = widgets.Output()
        self._output_tab = widgets.Tab(children=[self._output,
                                                 self._cluster_statistics_output
                                                 ])
        self._output_tab.set_title(0, 'ICE plots')
        self._output_tab.set_title(1, 'clustering')
        self._coord = widgets.Select(
            options=widget_repo.data._y_coord_names,
            value=widget_repo.data._y_coord_names[0],
            disabled=False
        )
        self._x_coord = widgets.Select(
            options=widget_repo.data._x_coord_names,
            value=widget_repo.data._x_coord_names[0],
            disabled=False
        )
        self._x_value_start = widgets.FloatText(value=-1.0)
        self._x_value_end = widgets.FloatText(value=1.0)
        self._n_x_points = widgets.IntText(value=10)
        self._accordion = widgets.Accordion(children=[
            self._get_selection_widget(),
            self._get_clustering_widget()
        ])

        self._accordion.set_title(0, 'Selection')
        self._accordion.set_title(1, 'Clustering')

    def _get_selection_widget(self):
        return widgets.VBox(children=[
            self._model_data_selection.get_widget(),
            widgets.Label(value='y-coordinates'),
            self._coord,
            widgets.Label(value='x-coordinates'),
            self._x_coord,
            widgets.Label(value='x-start'),
            self._x_value_start,
            widgets.Label(value='x-end'),
            self._x_value_end,
            widgets.Label(value='num x-points'),
            self._n_x_points,
            self._update_button])

    def _get_clustering_widget(self):
        self._update_clustering = widgets.Button(description='update')
        self._use_clustering = widgets.Checkbox(
            value=True, description='apply clustering')
        self._max_num_clusters = widgets.IntText(value=20,
                                                 description='maximum number of clusters')
        self._random_state = widgets.IntText(
            value=42, description='Random State')
        self._cache_in_repo = widgets.Checkbox(
            value=True, description='cache ICE in repo')
        self._scale = widgets.Checkbox(
            value=True, description='scale x-values to zero mean and unit variance')
        self._update_clustering.on_click(self._cluster)

        return widgets.VBox(children=[
            self._use_clustering,
            self._max_num_clusters,
            self._random_state,
            self._cache_in_repo,
            self._scale
        ])

    def _plot(self, d):
        cluster_param = None
        if self._use_clustering.value:
            cluster_param = {'n_clusters': self._max_num_clusters.value,
                             'random_state': self._random_state.value}
         # since the numpy cannot json serialized by default,
         # caching would not working, therefore we convert it into list
        x_points = [x for x in np.linspace(self._x_value_start.value, self._x_value_end.value,
                                           self._n_x_points.value)]
        self._ice = []
        for model, model_versions in self._model_data_selection.get_models().items():
            for data, data_versions in self._model_data_selection.get_data().items():
                for model_version in model_versions:
                    for data_version in data_versions:
                        self._ice.append((model, model_version, data, data_version,
                                          interpretation.compute_ice(widget_repo.ml_repo,
                                                                     x_points,
                                                                     data,
                                                                     model=model,
                                                                     model_version=model_version,
                                                                     data_version=data_version,
                                                                     y_coordinate=self._coord.value,
                                                                     x_coordinate=self._x_coord.value,
                                                                     cache=self._cache_in_repo.value,
                                                                     clustering_param=cluster_param,
                                                                     end_index=200),
                                          )
                                         )

        with self._output:
            clear_output(wait=True)
            display(go.FigureWidget(
                    paiplot.ice(self._ice)
                    ))
            self._output_tab.selected_index = 0
        if len(self._ice) > 0:
            if self._ice[0][-1].cluster_centers is not None:
                with self._cluster_statistics_output:
                    clear_output(wait=True)
                    display(go.FigureWidget(
                        paiplot.ice_clusters(self._ice)
                    ))

    def _cluster(self, d):

        self._output_tab.set_title(1, 'computing...')
        models = [x for x in self._models.value]

        for x in self._labels.value:
            l = widget_repo.labels[x]
            models.append((l['model'], l['version'],))

        with self._cluster_statistics_output:
            clear_output(wait=True)
            # display(self._recommendation_table.iloc[0:self._max_num_recommendations.value])
        self._output_tab.selected_index = 1
        self._output_tab.set_title(1, 'cluster statistics')
        # self._recommendation_selection.value = self._recommendation_table.index[0]

    @_add_title_and_border('Individual Conditional Expectation Plots')
    def get_widget(self):
        return widgets.HBox(children=[
            self._accordion,
            self._output_tab
        ])


class PlotMeasureVsParameter:
    def __init__(self):
        self._model_selector = widgets.Dropdown(
            options=widget_repo.model.get_models(), value=None)
        self._data_selector = _DataSelectorWithVersion(display_selection=False)
        # self._model_data_selector = _ModelAndDataSelectorWithVersion(
        #    display_selection=False)
        self._measure_selector = widgets.Dropdown(options=widget_repo.measures)
        self._model_selector.observe(
            self._update_param_selector)
        self._param_selector = widgets.Dropdown(options=[])
        self._output = widgets.Output()
        self._update_button = widgets.Button(description='update')
        self._update_button.on_click(self._plot)

    def _print(self, message):
        with self._output:
            clear_output(wait=True)
            print(message)

    def _update_param_selector(self, change):
        # print(change)
        if self._model_selector.value is None:
            return
        model = self._model_selector.value
        model_param_name = NamingConventions.get_model_param_name(
            model)
        params = []
        try:
            model_params = widget_repo.ml_repo.get(model_param_name)
            for p in model_params.get_params().keys():
                params.append(p)
        except:
            pass
        train_param_name = str(NamingConventions.TrainingParam(model))
        try:
            train_params = widget_repo.ml_repo.get(train_param_name)
            for p in train_params.get_params().keys():
                params.append(p)
        except:
            pass
        self._param_selector.options = params

    def _plot(self, change):
        measures = []
        model = self._model_selector.value
        if model is None:
            self._print('Please select a model.')
            return
        data = self._data_selector.get_data()
        for d, w in data.items():
            if len(w) > 0:
                measures.append(str(NamingConventions.Measure(
                    model=NamingConventions.get_model_from_name(model), data=d, measure_type=self._measure_selector.value)))
        if len(measures) == 0:
            self._print('Please select data together with data versions.')
            return
        with self._output:
            clear_output(wait=True)
            # create measure names from selected models, data and measures
            display(go.FigureWidget(
                    paiplot.measure_by_parameter(widget_repo.ml_repo,
                                                 measures, self._param_selector.value)
                    ))

    @_add_title_and_border('Measure vs Parameter')
    def get_widget(self):
        return widgets.HBox(children=[
            widgets.VBox(children=[
                widgets.VBox(children=[
                    widgets.Label(value='Model'),
                    self._model_selector]
                ),
                self._data_selector.get_widget(),
                self._measure_selector,
                self._param_selector,
                self._update_button
            ]),
            self._output
        ])
