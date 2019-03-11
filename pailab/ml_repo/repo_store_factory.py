
class RepoStoreFactory:

    @staticmethod
    def get_repo_stores():
        return ['disk_handler', 'git_handler', 'memory_handler']

    @staticmethod
    def get(repo_store_type, **kwargs):
        if repo_store_type == 'disk_handler':
            from pailab.ml_repo.disk_handler import RepoObjectDiskStorage
            return RepoObjectDiskStorage(**kwargs)
        elif repo_store_type == 'git_handler':
            from pailab.ml_repo.git_handler import RepoObjectGitStorage
            return RepoObjectGitStorage(**kwargs)
        elif repo_store_type == 'memory_handler':
            from pailab.ml_repo.memory_handler import RepoObjectMemoryStorage
            return RepoObjectMemoryStorage()
        raise Exception('Cannot create RepoStore: Unknown repo type ' + repo_store_type +
                        '. Use only types from the list returned by RepoStoreFactory.get_repo_stores().')


class NumpyStoreFactory:

    @staticmethod
    def get_numpy_stores():
        return ['memory_handler', 'hdf_handler']

    @staticmethod
    def get(numpy_store_type, **kwargs):
        if numpy_store_type == 'memory_handler':
            from pailab.ml_repo.memory_handler import NumpyMemoryStorage
            return NumpyMemoryStorage(**kwargs)
        elif numpy_store_type == 'hdf_handler':
            from pailab.ml_repo.numpy_handler_hdf import NumpyHDFStorage
            return NumpyHDFStorage(**kwargs)
        raise Exception('Cannot create NumpyStore: Unknown  type ' + numpy_store_type +
                        '. Use only types from the list returned by NumpyStoreFactory.get_numpy_stores().')
