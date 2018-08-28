import uuid
from enum import Enum
from repo.repo_objects import repo_object_init, RawData, RepoInfoKey  # pylint: disable=E0401
from repo.repo import MLObjectType  # pylint: disable=E0401


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

    def add(self, job):  # pragma: no cover
        """[summary]

        Arguments:
            job {[type]} -- [description]

        Returns:
            Id of job
        Raises:
            NotImplementedError -- [description]
        """
        raise NotImplementedError

    def get_status(self, jobid):  # pragma: no cover
        raise NotImplementedError

    def get_error_message(self, jobid):  # pragma: no cover
        raise NotImplementedError


class SimpleJobRunner:
    def __init__(self, repo):
        self._repo = repo
        self._job_status = {}
        self._error_message = {}

    def get_status(self, jobid):
        return self._job_status[jobid]

    def get_error_message(self, jobid):
        return self._error_message[jobid]

    def add(self, job):
        job_id = uuid.uuid1()
        self._job_status[job_id] = JobState.RUNNING.value
        try:
            job.run(self._repo, job_id)
        except Exception as e:
            self._job_status[job_id] = JobState.FAILED
            self._error_message[job_id] = str(e)
        self._job_status[job_id] = JobState.SUCCESSFULLY_FINISHED
