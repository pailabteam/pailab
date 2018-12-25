import logging
import collections

from pailab.repo import MLObjectType, MLRepo, NamingConventions
from pailab.repo import RawDataCollection,  TrainingDataCollection, TestDataCollection, ModelCollection  # pylint: disable=E0611, E0401
from pailab.repo_objects import RepoInfoKey  # pylint: disable=E0401
from pailab.repo_store import RepoStore  # pylint: disable=E0401
logger = logging.getLogger(__name__)


class MLTree:
    def __init__(self, ml_repo):
        self.raw_data = RawDataCollection(ml_repo)
        self.training_data = TrainingDataCollection(ml_repo)
        self.test_data = TestDataCollection(ml_repo)
        self.models = ModelCollection(ml_repo)
        #self.jobs = JobCollection(ml_repo)

    def modifications(self):
        result = {}
        result.update(self.raw_data.modifications())
        result.update(self.training_data.modifications())
        result.update(self.test_data.modifications())
        # result.update(self.models.modifications())
        return result
