Tutorial
===================


Creating a new repository
--------------------------

We first create a new repository for our task. The repository is the central key around all functionality is built.
Similar to a repository used for source control in classical software development it contains all data and algorithms needed for the machine 
learning task. The repository needs storages for 

- scripts containing the machine learning algorithms and interfaces,
- numerical objects such as arrays and matrices representing data, e.g. input data, data from the valuation of the models,
- json documents representing parameters, e.g. training parameter, model parameter.

To keep things simple, we just start using in memory storages. Note that the used memory interfaces are except for testing and playing around not be
the first choice, since when ending the session, everything will be lost...

In addition to the storages the repository needs a reference to a JobRunner which the platform can use to execute machine learning jobs. 
For this example we use the most simple one, executing everything sequential in the same thread, the repository runs in.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: creating new repository
    :end-before: end creating new repository


Adding training and testdata
------------------------------
The data in the repository is handled by two different data objects:

- :py:class:`pailab.repo_objects.RawData` is the object containing real data.
- :py:class:`pailab.repo.DataSet` is the object conaining the logical data, i.e. a reference to a RawData object together with a specification, which data from the RawData will be used. Here, one can specify a fixed version of the underlying RawData object (then changes to the RawData will not affect the derived DataSet) or a fixed or floating subset of the RawData by defininga start and endindex cutting the derived data just out of the original data.

Normally one will add RawData and then define DataSets which are used to train or test a model which is exactly the way shown in the following.

Adding RawData
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: add RawData 
    :end-before: end adding RawData


Adding DataSet
~~~~~~~~~~~~~~~~~~~~~~~~
.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: add DataSet
    :end-before: end adding DataSet

When creating the DataSet we have to set two important informations for the repository, given as a dictionary:

- The object name. Each object in the repository needs to have a unique name in the repository.
- The object type which gives. In our example here we say that we specify that the DataSet are training and test data. Note that on can have only one training data object pre repository while the repository can obtain many different test data sets.

Some may wonder what is now stored in *version_list*.
** Adding an object (independent if it is a data object or some other object such as a parameter), the object gets a 
version number and no object will be removed, adding just adds a new version.** The add method returns a dictionary of 
the object names together with their version number.


Adding a model
--------------------------
The next step to do machine learning would be to define a model which will be used in the repository. A model consists of the following pieces

- a skript where the code for the model valuation is defined together with the function name of the evaluation method
- a skript where the code for the model training is defined together with th function nam of the training method
- a model parameter object defining the model parameter and which must have implemented the correct interface so that it can be used within the repository (see the documentation on integrating new objects, normally there is not more to do then just simply add *@repo_object_init()* to the line above your *__init__* method)
- a training parameter object defining training parameters (such as number of optimization steps etc.), if necessary for your algorithms (this oen is optional)

To build and train models we use sklearn. In this case, we do not have to define the pieces defined above, since pailab provides a simple 
interface to sklearn defined in the module :py:mod:`pailab.externals.sklearn_interface`. This interface provides a simple method 
:py:meth:`pailab.externals.sklearn_interface.add_model`
to add an arbitrary sklearn model as a model which can be handled by the repository. This method adds a bunch of repo objects to the repository (according to the pieces described above):

- An object defining the function to be called to evaluate the model
- An object defining the function to be called to train the model
- An object defining the model
- An object defining the model parameter

In the following we just use a DecisionTree as our model.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: add model
    :end-before: end adding model

Train the model
-----------------------------------
Now, model taining is very simple, since you have defined training and testing data as well as  methods to value and fit your model and the model parameter.
So, you can just call :py:meth:`pailab.repo.MLRepo.run_training` on the repository, and the training is perfomred automatically.
The training job is executed via the JobRunner you specified setting up the repository. All method of the repository involving jobs return the job id when adding the job to the JobRunner so that you can control the status of the task and see if it sucessfully finished.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: run training
    :end-before: end running training

Define error measures and evaluate the model
----------------------------------------------
To measure errors and to provide plots the model must be evaluated on all test and training datasets. This can simply be accomplished by calling 
:py:meth:`pailab.repo.MLRepo.run_evaluation`.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: run evaluation
    :end-before: end running evaluation

Based on the evaluations one can specify different kinds of error measures. 

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: run measures
    :end-before: end running measures

Retrieving measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The measurement values are also stored as a repository object in the repository (see :py:class:`pailab.repo_objects.measure`). One can simply retrieve them by calling the repositories
:py:meth:`pailab.repo.MLRepo.get` method which can be used to retrieve all objects stored in the repository.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: get measures
    :end-before: end getting measures

Creating a list of all objects
----------------------------------
One can simply get an overview over all objects stored in the repository by calling :py:meth:`pailab.repo.MLRepo.get_names` to retrieve a list of names of 
all objects of a specific category (see :py:class:`pailab.repo.MLObjectType`). The following line will loop over all categories and print the names of all objects within this category
contained in the repository.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: list objects
    :end-before: end listing objects

