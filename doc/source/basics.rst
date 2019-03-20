Basics
======================



Core principles
------------------------
pailab's core is the :py:class:`pailab.ml_repo.repo.MLRepo` class which is what we call the machine learning repository.
The repository stores and versions all objects needed in the machine learning development cycle. 
There are three fundamental differences to version control systems such as git or svn for classical software development:

- Instead of source code, objects are checked into the repository. Here, each object must inherit or at least implement the respective methods from :py:class:`pailab.ml_repo.repo_objects.RepoObject` so that it can be handled by the repository. Furthermore, each such object belongs to a certain category (:py:class:`pailab.ml_repo.repo.MLObjectType`), so that the repository may perform certain checks and allow to automatize the ml build pipeline. 

- Each object is split into a part with standard data and a part with (large) numerical data and both parts are stored separately in different storages.
  Here, the normal data is stored in a storage derived from :py:class:`pailab.ml_repo.repo_store.RepoStore` whereas the numerical 
  data is stored via a :py:class:`pailab.ml_repo.repo_store.NumpyStore`.

- The execution of different jobs such as model training and evaluation or error computation is triggered via the MLRepo. 
  Here, the MLRepo simply uses a JobRunner to execute the jobs.

As we see, we need at least three ingredients to initialize an MLRepo:

- RepoStore
- NumpyStore
- JobHandler

In the next section we will show how to setup an MLRepo using these ingredients.

Setting up an MLRepo
------------------------------

In-memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The easiest way to start using pailab is instantiate MLRepo using all defaults, except the user which must be specified, otherwise an exception is thrown.

.. literalinclude:: ../../tests/repo_test.py
    :language: python
    :start-after: example with default
    :end-before: end example with default

This results in an MLRepo that handles everything in memory only, using  :py:class:`pailab.ml_repo.memory_handler.RepoObjectMemoryStorage` and :py:class:`pailab.ml_repo.memory_handler.NumpyMemoryStorage`
so that after closing the MLRepo, all data wil be lost. Therefore this should be only considered for testing or rapid and dirty prototyping. Note that in this case, the JobRunner 
used is the :py:class:`pailab.job_runner.job_runner.SimpleJobRunner` which simply runs all jobs sequential on the local machine in the same python thread the MLRepo has been constructed (synchronously).



Setup with git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this setion we briefly describe how to setup an MLRepo where the object data is stored in a git repository and the 
numpy data by a network or a shared drive like google drive.
So, let us assume that you have 
- a shared drive to store the numpy data, e.g. C:/shared_drive/repo_data,
- a centralized git repository.

To setup yor local repository you have to do the following steps.

- First clone a local repository from the centralized repo, let us assume that you do it into the folder C:/ml_repo/example_1/objects.
- Define the configuration settings for the repo as a python dictionary, e.g.

.. code-block:: python

    config = {
          'user': 'test_user',
          'workspace': 'c:/ml_repos/example_1',
          'repo_store': 
          {
              'type': 'git_handler',  
              'config': {
                  'folder': 'c:/ml_repos/example_1/objects', 
                  'file_format': 'json'
              }
          },
          'numpy_store':
          {
              'type': 'hdf_handler',
              'config':{
                  'folder': 'C:/shared_drive/repo_data',
                  'version_files': True
              }
          }
    }

Note that the workspace is used to store the configuration of the repository so that you must only specify the above settings in the setup and not in later use. 
The above configuration specifies the two main ingredient for the repository: The repo_store where all objects (without their numpy members ) are stored 
and the numpy_store for the numpy objects. Here we have chosen a git handled repo for the repo_store and an hdf_handler where all numpy objecs are stored
in hdf files. Using this configuration we can now init the ml repository.

.. code-block:: python

    from pailab import MLRepo
    ml_repo = MLRepo(config = config, save_config = True)

Note that you may from now on use the repository by just defining the workspace (all settings are stored in the .config.json file in the workspace):

.. code-block:: python

    ml_repo = MLRepo(workspace = 'c:/ml_repos/sc_new')

