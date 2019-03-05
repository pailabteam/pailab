Basics
======================

Setting up a local repository
------------------------------
In this setion we briefly describe how to checkout a local repository from a central git repo sharing the 
numpy data by a network drive or a shared drive like google drive.
So, let us assume that you have 
- a shared drive with numpy data, e.g. C:/shared_drive/repo_data,
- a centralized repo.

To setup yor local repository you have to do the follwoing steps.

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

