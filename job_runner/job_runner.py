import traceback
import datetime

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


class JobInfo:
    def __init__(self, user):
        self.user = user
        self.state = JobState.SUBMITTED.value
        self.submission_time = None
        self.error_message = None
        self.trace_back = None
        self.start_time = None
        self.end_time = None

    def set_state(self, state):
        self.state = state.value

    def set_start_time(self):
        self.start_time = datetime.datetime.now()

    def set_end_time(self):
        self.end_time = datetime.datetime.now()

    def __str__(self):
        result = self.user + ', ' + self.state + ', started ' + \
            str(self.start_time) + ', finished ' + str(self.end_time)
        if self.error_message is not None:
            result = result + ',  error_message: ' + self.error_message
        return result


class JobRunner(abc.ABC):
    """Baseclass for all job runners so that they can be used together with the MLRepo
    """

    @abc.abstractmethod
    def add(self, job, user):  # pragma: no cover
        """[summary]
        """
        pass

    @abc.abstractmethod
    def get_info(self, jobid):  # pragma: no cover
        """[summary]
        """
        pass


class SimpleJobRunner:
    def __init__(self, repo):
        self._repo = repo
        self._job_info = {}

    def set_repo(self, repo):
        self._repo = repo

    def get_info(self, jobid):
        return self._job_info[jobid]

    def add(self, job, user):
        job_id = uuid.uuid1()
        job_info = JobInfo(user)
        job_info.set_state(JobState.RUNNING)
        job_info.start_time = datetime.datetime.now()
        self._job_info[job_id] = job_info
        try:
            job.run(self._repo, job_id)
            job_info.end_time = datetime.datetime.now()
            job_info.set_state(JobState.SUCCESSFULLY_FINISHED)
        except Exception as e:
            job_info.end_time = datetime.datetime.now()
            job_info.set_state(JobState.FAILED)
            job_info.error_message = str(e)
            job_info.trace_back = traceback.format_exc()

        return job_id
