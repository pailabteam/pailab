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


Disk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To initialize an MLRepo so that the objects are stored on disk, we need to setup th respectiv storages within the MLRepo. 
One way to achieve this is to define the respective configurations in a dictionary and initialize the MLRepo with this dictionary.
An example is given 


.. literalinclude:: ../../tests/repo_test.py
    :language: python
    :start-after: diskhandlerconfig
    :end-before: end diskhandlerconfig

First we see that there is a user and also a workspace defined in the dictionary. The workspace is a directory where the configuration and settings are stored so that when you
instantiate the MLRepo again, you just need to specify the workspace and not the whole settings again.
The RepoStore used within the MLRepo is defined via the dictionary belonging to the repo_store key. Here we see that the configuration consists of describin the type of store
(here we use the disk_handler which simply stores the objects on disk) and the settings for this storage. In our example the objects are stored in json format in the 
folder example_1/objects.
The NumpyStore internally used is selected so that the big data will be stored in hdf5 files.

Now we simply instantiate the MLRepo using this configuration.


.. literalinclude:: ../../tests/repo_test.py
    :language: python
    :start-after: instantiate diskhandler
    :end-before: end instantiate diskhandler

To instantiate the MLRepo and directly save the respective config you have to set the parameter save_config

.. literalinclude:: ../../tests/repo_test.py
    :language: python
    :start-after: instantiate diskhandler save config
    :end-before: end instantiate diskhandler save config

Saving the config you may instantiate the MLRepo another time simply by 

.. literalinclude:: ../../tests/repo_test.py
    :language: python
    :start-after: instantiate with workspace
    :end-before: end instantiate with workspace



git
~~~~~~~~~~~~~~~~~~~~~~
The previous example stored the objects simply as json files on disk. There is the possibility to use git to manage the files. Here, you just have to replace the type by 'git_handler',
i.e. change the type simply to git_handler.

If you have a remote git repository which you want to use as a remote, you have to clone the repository first and then specify the directory of the cloned repo 
as directory of the git_handler.
