class JobRunnerFactory:

    @staticmethod
    def get_job_runners():
        return ['simple', 'sqlite']

    @staticmethod
    def get(job_runner_type, repo, **kwargs):
        if job_runner_type == 'simple':
            from pailab.job_runner.job_runner import SimpleJobRunner
            return SimpleJobRunner(repo=repo, **kwargs)
        elif job_runner_type == 'sqlite':
            from pailab.job_runner.job_runner import SQLiteJobRunner
            return SQLiteJobRunner(repo=repo, **kwargs)
        raise Exception('Cannot create JobRunner: Unknown JobRunner type ' + job_runner_type +
                        '. Use only types from the list returned by JobRunnerFactory.get_job_runners().')
