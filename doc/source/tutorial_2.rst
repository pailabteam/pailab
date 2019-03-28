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



