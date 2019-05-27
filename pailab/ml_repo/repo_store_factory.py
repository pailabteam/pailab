
class RepoStoreFactory:
    """ the repo stores factory

    The factory is used to create Repo handlers

    """

    @staticmethod
    def get_repo_stores():
        """ return a list of supported repo handlers

        Returns:
            list of str -- the list of supported handlers
        """

        return ['disk_handler', 'git_handler', 'memory_handler']

    @staticmethod
    def get(repo_store_type, **kwargs):
        """ this method returns a handler of the specified type

        This method is used to construct handler of the specified type using the provided arguments.
        Currently these types are supported:
        * disk_handler
        * git_handler
        * memory_handler

        Arguments:
            repo_store_type {str} -- the name of the repo store type

        Raises:
            Exception -- raises an exception if the type is not supported

        Returns:
            RepoStore -- the constructed handler
        """

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
    """ class to construct a numpy store for big data
    """

    @staticmethod
    def get_numpy_stores():
        """ this method returns the supported types

        Returns:
            list of str -- list of supported big data handler types
        """

        return ['memory_handler', 'hdf_handler', 'hdf_remote_handler']

    @staticmethod
    def get(numpy_store_type, **kwargs):
        """ this method returns a handler of the specified type for handling big data

        This method is used to construct a big data handler of the specified type using the provided arguments.
        Currently these types are supported:
        * memory_handler
        * hdf_handler

        Arguments:
            numpy_store_type {str} -- the name of the big data store type

        Raises:
            Exception -- raises an exception if the type is not supported

        Returns:
            NumpyStore -- the constructed handler
        """
        if numpy_store_type == 'memory_handler':
            from pailab.ml_repo.memory_handler import NumpyMemoryStorage
            return NumpyMemoryStorage(**kwargs)
        elif numpy_store_type == 'hdf_handler':
            from pailab.ml_repo.numpy_handler_hdf import NumpyHDFStorage
            return NumpyHDFStorage(**kwargs)
        elif numpy_store_type == 'hdf_remote_handler':
            from pailab.ml_repo.numpy_handler_hdf import NumpyHDFRemoteStorage
            return NumpyHDFRemoteStorage(**kwargs)
        raise Exception('Cannot create NumpyStore: Unknown  type ' + numpy_store_type +
                        '. Use only types from the list returned by NumpyStoreFactory.get_numpy_stores().')
