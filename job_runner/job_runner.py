import traceback
import traceback

import abc

import uuid
from enum import Enum
from repo.repo_objects import repo_object_init, RawData, RepoInfoKey  # pylint: disable=E0401
from repo.repo import MLObjectType  # pylint: disable=E0401


class Job(abc.ABC):
    """Abstract class defining the interfaces needed for a job to b used in the JobRunner

    """
    @abc.abstractmethod
    def get_predecessor_jobs(self):
        """Return list of jobids which must have been run sucessfully before the job will be executed
        """
        pass

    @abc.abstractmethod
    def run(self, ml_repo, job_id):
        pass


class JobState(Enum):
    """Job states
    """
    SUBMITTED = 'submitted'
    RUNNING = 'running'
    SUCCESSFULLY_FINISHED = 'successfully_finished'
    FAILED = 'failed'


class JobRunner(abc.ABC):
    """Baseclass for all job runners so that they can be used together with the MLRepo
    """

    @abc.abstractmethod
    def add(self, job):  # pragma: no cover
        """[summary]

        Arguments:
            job {[type]} -- [description]

        Returns:
            Id of job
        Raises:
            NotImplementedError -- [description]
        """
        pass

    @abc.abstractmethod
    def get_status(self, jobid):  # pragma: no cover
        pass

    @abc.abstractmethod
    def get_error_message(self, jobid):  # pragma: no cover
        pass

    @abc.abstractmethod
    def get_trace_back(self, jobid):  # pragma: no cover
        pass


class SimpleJobRunner:
    def __init__(self, repo):
        self._repo = repo
        self._job_status = {}
        self._error_message = {}
        self._trace_back = {}

    def get_status(self, jobid):
        return self._job_status[jobid]

    def get_error_message(self, jobid):
        return self._error_message[jobid]

    def get_trace_back(self, jobid):  # pragma: no cover
        return self._trace_back[jobid]

    def add(self, job):
        job_id = uuid.uuid1()
        self._job_status[job_id] = JobState.RUNNING.value
        try:
            job.run(self._repo, job_id)
            self._job_status[job_id] = JobState.SUCCESSFULLY_FINISHED
        except Exception as e:
            self._trace_back[job_id] = traceback.format_exc()
            self._job_status[job_id] = JobState.FAILED
            self._error_message[job_id] = str(e)

        return job_id
