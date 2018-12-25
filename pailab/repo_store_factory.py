
class RepoStoreFactory:

    @staticmethod
    def get_repo_stores():
        return ['disk_handler', 'git_handler', 'memory_handler']

    @staticmethod
    def get(repo_store_type, **kwargs):
        if repo_store_type == 'disk_handler':
            from pailab.disk_handler import RepoObjectDiskStorage
            return RepoObjectDiskStorage(**kwargs)
        elif repo_store_type == 'git_handler':
            from pailab.git_handler import RepoObjectGitStorage
            return RepoObjectGitStorage(**kwargs)
        elif repo_store_type == 'memory_handler':
            from pailab.memory_handler import RepoObjectMemoryStorage
            return RepoObjectMemoryStorage()
        raise Exception('Cannot create RepoStore: Unknown repo type ' + repo_store_type +
                        '. Use only types from the list returned by RepoStoreFactory.get_repo_stores().')
