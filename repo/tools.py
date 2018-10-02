import logging
from repo.repo import MLObjectType, MLRepo  # pylint: disable=E0611, E0401
from repo.repo_objects import RepoInfoKey  # pylint: disable=E0401
from repo.repo_store import RepoStore  # pylint: disable=E0401
logger = logging.getLogger('repo.analyzer')


def check_model(repo, model_name, correct=False, model_version=RepoStore.LAST_VERSION):
    """Check if the model is calibrated and evaluated on the latest versions

    Arguments:
        repo {MLRepo} -- the ml repo
        model_name {str} -- model name

    Keyword Arguments:
        correct {bool} -- determine whether training, evaluations, measures and tests will be qutomatically triggered if check fails to correct (default: {False})
        model_version {repo_version} --  (default: {RepoStore.LAST_VERSION})

    Returns:
        dict -- dictionary containing the modifier versions and the latest version of the objects, if dictionary is empty noc model inconsistencies could be found
    """

    result = {}

    m = repo._get(MLRepo.get_calibrated_model_name(
        model_name), version=model_version)
    # first check if all versions of the models modifiers are still the latest version
    repo_store = repo.get_ml_repo_store()
    for k, v in m.repo_info[RepoInfoKey.MODIFICATION_INFO].items():
        latest_version = repo_store.get_latest_version(k)
        if not v == latest_version:
            result[k] = {'modifier version': v,
                         'latest version': latest_version}
    if len(result) > 0 and correct == True:
        job_id = repo.run_training(model_name)
        job_ids = repo.run_evaluation(
            model_name, message='running evaluations triggered by new training of ' + model_name, predecessors=[job_id])
        job_ids = repo.run_measures(
            model_name, message='running measures triggered by new training of ' + model_name, predecessors=job_ids)
        job_ids = repo.run_tests(
            model_name, message='running tests triggered by new training of ' + model_name, predecessors=job_ids)
    return result


def _check_evaluation(repo, model, data):
    pass


def check_evaluation(repo):
    """Checks if all defined evaluations in the repo are on latest model and datasets


    Arguments:
        repo {MLRepo} -- the repository the checks are applied to
    """
    logger.info('Start checking evaluations.')
