Labeling, testing, consistency
=========================================

Labeling model versions
-----------------------------------
The ``MLRepo`` offers the possibility to label a certain model version. This gives the user the possibility to mark 
certain models, e.g. labeling the model that goes into production or labeling the model which has the best error measure
as a candidate for a future release. Labels are not only just nice to identify certain models more easily than remembering 
the version number, they are also supported by other methods form pailab:  As we will see in this tutorial, consistency checks
are applied to all labeled method, regression tests may be defined using labeled models or  plotting methods will include the labeled
models and explicitly highlight them.

Setting a label is quite simple using the :py:meth:`pailab.MLRepo.set_label` method

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: label snippet
    :end-before: end label snippet

Note that we have used the ``LAST_VERSION`` keyword. Instead of specifying the exact version string, nearly all methods who need a version
do also accept the ``LAST_VERSION`` and ``FIRST_VERSION`` keywords.

The ``set_label`` method creates just an object of :py:class:`pailab.repo_objects.Label` and stores it in the repository. Therefore listing all labels 
in the repo can be performed by using ``get_names`` again::

    >>print(ml_repo.get_names(MLObjectType.LABEL))
    ['prod']

We can see what model and model version a label refers to by just getting the label object and checking the ``name`` and ``version`` attributes::

    >>label = ml_repo.get('prod')
    >>print(label.name)
    >>print(label.version)
    DecisionTreeRegressor/model
    69c86a46-512a-11e9-b7bd-fc084a6691eb

Automated testing
-----------------------------------

There is a lot of debate whether unit testing or regression testing would make sense for ML. However, everyone should decide on his own
for his project if it would make sense for his problems or not and pailab supports automated testing for those who want to apply it.

A test basically consists of two parts: 

- A definition of the principal test containing the type of test and a definition for what data and models the tests are created
- the tests itself which are also jobs executed by the ``MLRepo``'s internal JobRunner

We define a set of regression tests using :py:class:`pailab.tools.tests.RegressionTestDefinition`.
Here, pailab's  RegressionTest compares specified error measures of a model with error measures of a 
reference model (typically the one in production, maybe labeled 'prod' ;-) )

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after:  test definition snippet
    :end-before: end test definition snippet

We may run the test by calling :py:meth:`pailab.repo.MLRepo.run_test`

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after:  add test snippet
    :end-before: end add test snippet

where ``tests`` is a list of tuples, each containing the name of the test as well as the respective version::

    >>print(tests)
    [('DecisionTreeRegressor/tests/reg_test/test_data', '5b71ad5a-516f-11e9-bf7c-fc084a6691eb'), 
    ('DecisionTreeRegressor/tests/reg_test/training_data', '5b8b46ca-516f-11e9-990d-fc084a6691eb')]

The attribute ``result`` of the test object contains the result of the test (if it was successful or not)::

    >>test = ml_repo.get('DecisionTreeRegressor/tests/reg_test/test_data')
    >>print(test.result)
    'succeeded'
    
Consistency checks
-----------------------------------
Pailab's :py:mod:`pailab.tools.checker` -submodule provides functionality to check for consistency and
quality issues as well as for outstanding tasks (such as rerunning a training after the training set has been changed).


Model consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are different checks to test model consistency such as if the tests of a model are up to date and succeeded or if the 
latest model is trained on the latest training data. All model tests are performed for labeled models and the latest model only.

The following checks are performed:
- Is the latest model calibrated on the latest parameters and training data
- Are all labeled models (including latest model) evaluated on the latest available training and test data
- Are all measures of all labeled models computed on the latest data
- Have all tests been run on the labeled models

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after:  run check snippet
    :end-before: end run check snippet

The variable ``inconsistencies`` contains all inconsistencies found which means in our case that the list is empty::

    >>print(inconsistencies)
    []

Now we change a model parameter but do not start a new training

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: add inconsistency snippet
    :end-before: end add inconsistency snippet

Running the consistency check again leads to::

    >>print(inconsistencies)
    :undefined
[{'DecisionTreeRegressor/model:last': {'latest model version not on latest inputs': 
    {'DecisionTreeRegressor/model_param': {'modifier version': 'cdc3fed4-5192-11e9-a7fd-fc084a6691eb', 
            'latest version': 'cfe1b9fa-5192-11e9-b360-fc084a6691eb'}}}}]
