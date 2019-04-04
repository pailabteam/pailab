.. _tutorial:

Repo initialization, training, evaluation
=========================================


Creating a new repository
--------------------------

We first create a new repository for our task. The repository is the central key around all functionality is built.
Similar to a repository used for source control in classical software development it contains all data and algorithms needed for the machine 
learning task. The repository needs storages for 

- numerical objects such as arrays and matrices representing data, e.g. the input data or data from the valuation of the models
- small objects (of part of objects after cutting out the numerical objects), e.g. training parameter, model parameter.

To keep things simple, we may simply use the default constructor of the ``MLRepo`` creating in memory storages. 

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: creating in memory storage
    :end-before: end creating in memory storage


Note that the memory interfaces used in this tutorial are 
useful for testing or playing around but may not be your choice for real life applications 
(except that you are willing to start your work again after your computer has been rebooted :-) ).
In this case, you could use a simple storage using json format to store small data in files while using a 
storage saving the numpy data in hdf5 files. In this case you have to specify this in a respective configuration dictionary.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: creating new repository
    :end-before: end creating new repository


In addition to the storages the repository needs a reference to a ``JobRunner`` which the platform can use to execute different jobs needed during
your ML development process. As long as we do not specify another ``JobRunnner``, the ``MLRepo`` uses the most simple 
:py:class:`pailab.job_runner.job_runner.SimpleJobRunner` as default, 
that executes everything sequential in the same thread the repository runs in. There are two possibilities to set the ``JobRunner``. You may use the 
configuration settings as shown above. In this case, the :py:class:`pailab.job_runner.job_runner_factory.JobRunnerFactory`is used to create the 
respective `JobRunner` within `MLRepo`'s constructor. Another possibility you may use (e.g. if you implemented your on JobRunner and you do not want to integrate
it into the factory), you may simply instantiate the respective ``JobRunner`` and set it into the MLRepo's ``job_runner`` attribute

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: specifying job runner
    :end-before: end specifying job runner

.. NOTE::

    The ''MLRepo'' uses the :py:class:`pailab.job_runner.job_runner.SimpleJobRunner` as default, you do only have to set a ``JobRunner`` as shown above if you want to use
    a different one.

Adding training and test data
------------------------------
The data in the repository is handled by two different data objects:

- :py:class:`pailab.ml_repo.repo_objects.RawData` is the object containing real data.
- :py:class:`pailab.ml_repo.repo.DataSet` is the object containing the logical data, i.e. a reference to a RawData object together with a specification, 
  which data from the RawData will be used. Here, one can specify a fixed version of
  the underlying RawData object (then changes to the RawData will not affect the derived DataSet) or a fixed or floating subset of the RawData by 
  defining start and end index cutting the derived data just out of the original data.

Normally, for training and testing we will use ``DataSet``. So, we first have to add the data in form of a ``RawData`` object and then define the respective
DataSets based on this RawData.



Adding RawData
~~~~~~~~~~~~~~~~~~~~~~
We first read the data from a csv file using pandas.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: read pandas
    :end-before: end read pandas

Now ``data`` holds a pandas dataframe and we have to extract the respective x and y values as numpy matrices
to use them to create the ``RawData`` object.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: extract data
    :end-before: end extract data

Using the numpy objects we can now create the ``RawData`` object and add it to the repo.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: add RawData snippet
    :end-before: end adding RawData snippet

Adding DataSet
~~~~~~~~~~~~~~~~~~~~~~~~
Now, base on the RawData, we can add the training and test data sets.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: add DataSet
    :end-before: end adding DataSet

.. NOTE::

    We have to define the type of object via setting a value from :py:class:``pailab.repo.MLObjectType`` for the ``RepoInfoKey.CATEGORY`` key object. The
    category is used by the MLRepo to support certain automatizations and checks. 

The ``version_list`` variable is a dictionary that maps the object names of the added objects to their version. 

Adding a model
--------------------------
The next step to do machine learning would be to define a model which will be used in the repository. A model consists of the following pieces

- a function for evaluating the model
- a function for model training
- a model parameter object holding the model parameters
- a training parameter object defining training parameters

We would like to use the DecisionTreeRegressor from sklearn in our example below.
In this case, we do not have to define the pieces defined above, since pailab provides a simple 
interface to sklearn defined in the module :py:mod:`pailab.externals.sklearn_interface`. This interface provides a method 
:py:meth:`pailab.externals.sklearn_interface.add_model`
to add an arbitrary sklearn model as a model which can be handled by the repository. 
The method :py:meth:`pailab.externals.sklearn_interface.add_model` 
creates internally a :py:class:`pailab.repo_objects.Model` object defining the objects listed above and adds it to the repository. 
We refer to :ref:`adding_model` for details on setting up the model object
and to :ref:`integrating_model`for details how to integrate your own algorithm or other external ml platforms.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: add model
    :end-before: end adding model

Train the model
-----------------------------------
Now, model training is very simple, since you have defined training and testing data as well as  
methods to value and fit your model and the model parameter.
So, you can just call :py:meth:`pailab.ml_repo.repo.MLRepo.run_training` 
on the repository, and the training is performed automatically.
The training job is executed via the JobRunner you specified setting up the repository. 
All method of the repository involving jobs return the job id when adding the job to the JobRunner so that you can control 
the status of the task and see if it successfully finished.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: run training
    :end-before: end running training

The variable ``job_id`` contains  a tuple with the id the job is stored in the repo and the respective version::

    >> print(job_id)
    ('DecisionTreeRegressor/jobs/training', '69c7ce4a-512a-11e9-ab9f-fc084a6691eb')

This information can be used to retrieve the underlying job object. The job object contains certain useful information such as the status 
of the job, i.e. if it is waiting, running or if it has been finished, the time the job has been started 
or messages of errors that occurred during execution::

    >>job = ml_repo.get(job_id[0], job_id[1])
    >>print(job.state)
    finished
    >>print(job.started)
    2100-03-28 08:23:41.668922

.. NOTE::

    The jobs are only executed if they have not yet been run on the input. So that if we call ``run_training`` again, we get 
    a message that the job has already been run::
    
        >> job_id = ml_repo.run_training() 
        >> print(job_id)
        No new training started: A model has already been trained on the latest data.

We can check that the training was successful by checking whether a calibrated object for the specified model has been created.
For this, we simply list all object names of objects from the category ``MLObjectType.CALIBRATED_MODEL`` stored within the repo
using the :py:meth:``pailab.repo.MLRepo.get_names`` method::

    >>print(ml_repo.get_names(MLObjectType.CALIBRATED_MODEL))
    ['DecisionTreeRegressor/model']

As we see, an object with name ``'DecisionTreeRegressor/model'`` has been created and stored in the repo.



Model evaluation and error measurement
----------------------------------------------

Evaluate a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To measure errors and to provide plots the model must be evaluated on all test and training datasets. This can simply be accomplished by calling 
:py:meth:`pailab.ml_repo.repo.MLRepo.run_evaluation`.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: run evaluation
    :end-before: end running evaluation

This method has now applied the model's evaluation method to all test and training 
data stored in the repository and also stored the results. Similar to the model training we may list all results 
using the ``get_names`` method::

    >>print(ml_repo.get_names(MLObjectType.EVAL_DATA))
    ['DecisionTreeRegressor/eval/sample2', 'DecisionTreeRegressor/eval/sample1']

As we see, we have two different objects containing the evaluation of the model, one for each dataset stored.
Note that we can check what model and data has been used to create these evaluations. We just have to 
look at the ``modification_info`` attribute of the ``repo_info`` data attached to each object stored in the ``MLRepo``::

    >>eval_data = ml_repo.get('DecisionTreeRegressor/eval/sample2')
    >>print(eval_data.repo_info.modification_info)
    {'DecisionTreeRegressor/model': '69c86a46-512a-11e9-b7bd-fc084a6691eb', 
    'DecisionTreeRegressor': '687c5da8-512a-11e9-b0b4-fc084a6691eb', 
    'sample2': '6554763b-512a-11e9-938e-fc084a6691eb', 
    'eval_sklearn': '687bc058-512a-11e9-8b3e-fc084a6691eb', 
    'DecisionTreeRegressor/model_param': '687c5da7-512a-11e9-99c4-fc084a6691eb'}

The ``modification_info`` attribute  is a dictionary 
that maps all objects involved in the creation of the respective object to their version that has been used to derive the object. 
We can directly see the versions of the calibrated model ``'DecisionTreeRegressor/model'`` as well as the version of the underlying data set
``'sample2'``. 

Define error measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now we may add certain error measures to the repository

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: add measures snippet
    :end-before: end add measure snippet

which can be evaluated by 

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: run measures snippet
    :end-before: end run measures snippet

As before, we get an overview of all measures computed and stored in the repository (as a repo object, see :py:class:`pailab.ml_repo.repo_objects.measure`)
using the ``get_names`` method::

    >>print(ml_repo.get_names(MLObjectType.MEASURE))
    ['DecisionTreeRegressor/measure/test_data/max', 
    'DecisionTreeRegressor/measure/test_data/r2', 
    'DecisionTreeRegressor/measure/training_data/max', 
    'DecisionTreeRegressor/measure/training_data/r2']


Retrieving measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The computed value is stored in the measurement object in the attribute value::

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: get measures
    :end-before: end getting measures

prints the value of the measurement.

Creating a list of all objects
----------------------------------
One can simply get an overview over all objects stored in the repository by calling :py:meth:`pailab.ml_repo.repo.MLRepo.get_names` to retrieve a list of names of 
all objects of a specific category (see :py:class:`pailab.ml_repo.repo.MLObjectType`). The following line will loop over all categories and print 
the names of all objects within this category
contained in the repository.

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: list objects
    :end-before: end listing objects
