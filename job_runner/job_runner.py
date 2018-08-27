import uuid
from enum import Enum
from repo.repo_objects import repo_object_init, RawData, RepoInfoKey  # pylint: disable=E0401
from repo.repo import MLObjectType


class JobState(Enum):
    """Job states
    """
    SUBMITTED = 'submitted'
    RUNNING = 'running'
    SUCCESSFULLY_FINISHED = 'successfully_finished'
    FAILED = 'failed'


class JobRunner:
    """Baseclass for all job runners so that they can be used together with the MLRepo
    """

    def add(self, job):
        """[summary]

        Arguments:
            job {[type]} -- [description]

        Returns:
            Id of job
        Raises:
            NotImplementedError -- [description]
        """
        raise NotImplementedError

    def get_status(self, jobid):
        raise NotImplementedError

    def get_error_message(self, jobid):
        raise NotImplementedError


class SimpleJobRunner:
    def __init_(self, repo):
        self._repo = repo
        self.job_status = {}
        self.error_message = {}

    def get_status(self, jobid):
        return self.job_status[jobid]

    def get_error_message(self, jobid):
        return self.error_message[jobid]

    def add(self, job):
        job_id = uuid.uuid1()
        self.job_status[job_id] = JobState.RUNNING.value
        try:
            job.run(self._repo, job_id)
        except Exception as e:
            self.job_status[job_id] = JobState.FAILED
            self.error_message[job_id] = str(e)
        self.job_status[job_id] = JobState.SUCCESSFULLY_FINISHED
