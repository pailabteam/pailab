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

The variable ``inconsistencies`` contains a list of all inconsistencies found. In our case the list is currently empty since there
are no inconsistencies::

    >>print(inconsistencies)
    []

Now we change a model parameter but do not start a new training

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: add inconsistency snippet
    :end-before: end add inconsistency snippet

We run the consistency check again::

    >>print(inconsistencies)
    [{'DecisionTreeRegressor/model:last': {'latest model version not on latest inputs': 
    {'DecisionTreeRegressor/model_param': {'modifier version': 'cdc3fed4-5192-11e9-a7fd-fc084a6691eb', 
            'latest version': 'cfe1b9fa-5192-11e9-b360-fc084a6691eb'}}}}]

Now we get a list containing one dictionary that contains the model inconsistencies. In our case, the dictionary shows
one inconsistency: There are model inputs the latest  calibrated model of ``'DecisionTreeRegressor/model'`` 
has not yet been calibrated on. It also shows us that the model parameter ``'DecisionTreeRegressor/model_param'`` is the input that
is newer then the one used in the latest version.

We can fix this issue by running a new training::

    >>ml_repo.run_training()

Rerun training fixes the training but leads to new problems. Now after having retrained,
the evaluation of the new model on the data sets as well as the computation of the defined error measures 
are now missing::

        >>print(inconsistencies)
        [{'DecisionTreeRegressor/model:last': {
            'evaluations missing': {
                'training_data': '429f88ba-524d-11e9-98f6-fc084a6691eb', 
                'test_data': '429f88ba-524d-11e9-98f6-fc084a6691eb'}, 
            'measures not calculated': 
                {
                'DecisionTreeRegressor/measure/training_data/max', 
                'DecisionTreeRegressor/measure/test_data/r2', 
                'DecisionTreeRegressor/measure/test_data/max', 
                'DecisionTreeRegressor/measure/training_data/r2'}}}]

Now we may fix these issues by calling first :py:meth:`pailab.repo.MLRepo.run_evaluation` and then 
:py:meth:`pailab.repo.MLRepo.run_measures` or we can simply call ``run_evaluation`` only, 
setting the parameter ``run_descendants`` to ``True``. By doing so, the ``MLRepo`` resolves all steps of the build pipeline 
following the model evaluation 

    >>print(checker.run(ml_repo))
    []

Training and test data consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pailab does also perform checks w.r.t. the training and test data. Here, one check is if test and training data overlap.
To illustrate this, we add a second test data set to the repo which overlaps with the training data. Note that we first run 
the evaluation on the new data set so that we do not see again the errors that evaluation or error measures are missing for this data

.. literalinclude:: ../../tests/tutorial_test.py
    :language: python
    :start-after: add second test data snippet
    :end-before: end add second test data snippet

Now, performing the check shows a lot of inconsistencies  a check::

    >>print(checker.run(ml_repo))
    [{'test_data_2': {'training and test data overlap': {'test_data_2': 'ecdc36ee-5465-11e9-92e2-fc084a6691eb', 'training_data': 'e6ac4eba-5465-11e9-b956-fc084a6691eb'}}}]

Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We may also check the overall test status. Here we have to call :py:meth:`pailab.tools.checker.Tests.run`::

    >>print(checker.Tests.run(ml_repo))
    {'DecisionTreeRegressor/model:323c05e8-5483-11e9-88ea-fc084a6691eb': 
        {'DecisionTreeRegressor/tests/reg_test/test_data': 
            'Test for model DecisionTreeRegressor/model, 
            version 323c05e8-5483-11e9-88ea-fc084a6691eb on latest data test_data missing.', 
        'DecisionTreeRegressor/tests/reg_test/test_data_2': 
            'Test for model DecisionTreeRegressor/model, version 323c05e8-5483-11e9-88ea-fc084a6691eb on latest data test_data_2 missing.', 
        'DecisionTreeRegressor/tests/reg_test/training_data': 
            'Test for model DecisionTreeRegressor/model, version 323c05e8-5483-11e9-88ea-fc084a6691eb on latest data training_data missing.'
        }, 
    'DecisionTreeRegressor/model:prod': 
        {'DecisionTreeRegressor/tests/reg_test/test_data_2': 
            'Test for model DecisionTreeRegressor/model, version 301bcc1e-5483-11e9-82a2-fc084a6691eb on latest data test_data_2 missing.'
        }
    }

