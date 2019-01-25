from ipywidgets import *
import ipywidgets as widgets
from IPython.display import display, clear_output

from pailab import MLObjectType, RepoInfoKey


class ObjectOverviewList:
    def __init__(self, ml_repo):
        self._ml_repo = ml_repo
        self._categories = widgets.SelectMultiple(
            options=[k.value for k in MLObjectType])
        self._repo_info = widgets.SelectMultiple(
            options=[k.value for k in RepoInfoKey]
        )
        self._button_update = widgets.Button(description='update')
        self._button_update.on_click(self.get_overview)
        self.output = widgets.Output()
        self._input_box = widgets.HBox(
            children=[widgets.VBox(
                children=[self._categories, self._repo_info]), self._button_update, self.output]
        )

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
        for k in self._categories.value:
            names = self._ml_repo.get_names(k)
            for n in names:
                print(n + '\t  ' + k.value)
        pass

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
        #self.cb_additional_fields = []
        # return self.final_widget
