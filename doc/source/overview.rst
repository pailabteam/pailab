

Overview
-------------------------------
Source code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pailab's source code is available on GitHub:

https://github.com/pailabteam/pailab

and cloned using::

    git clone https://github.com/pailabteam/pailab.git pailab

Examples and first steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To learn how to work with pailab you may find the :ref:`overall_tutorial` useful. The tutorial gives an introduction to all building blocks and tools.

Notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you like working with jupyter, there are the following jupyter notebooks demonstrating pailab's functionality. 
The notebooks are located in the  
`examples <https://github.com/pailabteam/pailab/tree/develop/examples>`_ folder of pailab's github repo.
Please note that the plots in these notebooks are created using plotly. Therefore if you want to play around with the 
plotting functionality you have to install this. However, even if you do not want to install plotly, the notebooks are nevertheless 
a very good starting point.

**Introductionary**

    - `boston_housing.ipynb`_: Shows pailab's basic functionality using the boston housing data  set and a regression tree from scikit learn (without preprocessing)
    - `adult-census-income.ipynb`_: Shows pailab's basic functionality using the adult-census data set and a regression tree from scikit learn (including preprocessing)
    - `boston_housing_widgets.ipynb`_:  Similar to boston_housing.ipynb but using some jupyter widgets
    - `boston_housing_distributed.ipynb`_: Similar to boston_housing.ipynb but using the :py:class:`pailab.job_runner.SQLiteJobRunner` job runner to execute jobs in a different thread or even on a different machine
    
**Advanced**
    
    - `caching_demo.ipynb`_: Simple example demonstrating the caching of results of time consuming functions.

    .. _boston_housing.ipynb: https://nbviewer.jupyter.org/github/pailabteam/pailab/blob/develop/examples/boston_housing/boston_housing.ipynb
    .. _adult-census-income.ipynb: https://nbviewer.jupyter.org/github/pailabteam/pailab/blob/develop/examples/adult-census-income/adult-census-income.ipynb
    .. _boston_housing_widgets.ipynb: https://nbviewer.jupyter.org/github/pailabteam/pailab/blob/develop/examples/boston_housing/boston_housing_widgets.ipynb
    .. _boston_housing_distributed.ipynb: https://nbviewer.jupyter.org/github/pailabteam/pailab/blob/develop/examples/boston_housing/boston_housing_distributed.ipynb
    .. _caching_demo.ipynb: https://nbviewer.jupyter.org/github/pailabteam/pailab/blob/develop/examples/caching_demo.ipynb

Logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pailab uses the Python standard logging module :mod:`logging` making use of different 
log levels. The log level of the logger may be globally set by::

    import logging as logging
    logging.basicConfig(level=logging.DEBUG)

See the ``logging`` modules documentation for further details.


.. |laby| image:: images/alien.png
    :height: 180
    :alt: Laby

.. |bugy| image:: images/monster.png
    :height: 90
    :alt: Bugy

Laby and Bugy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
During a trans-universal trip in 2500 Laby and Bugy made a stop on earth.
They were quite astonished to see how far humans 
had developed the AI business but they got a little frightened when they saw how blind-folded humans worked in this business. At least Laby would have not been 
far from a heart attack if he would have had a thing we humans might call heart. They soon decided to help these poor underdeveloped 
human species and to make a little time travel to the beginning of the AI bubble. So, when they arrived in January 2019, they started to 
develop pailab. 

They form quite a good team, complementing each other. Laby fights each bug with her weapons brought from her home planet Labmania, 
documenting everything (Bugy had to put a lot of effort convincing her not to document the documentation) and producing code
passing all beautifiers for all styleguids without being changed (and has overruled Chuck Norris since his code was not matlab compliant). 
She is really enthusiastic about testing.

Bugy is more the chaotic but creative alien. He loves to produce many new functionalities and hates documenting. He has not understand the
sense of measuring test coverage but he may implement a better Powerpoint application then the one implemented by his cousin Bugsy 
(who uses the initials MS for whatever reason) with maybe three lines of code 
(if we count his comment to understand his program the source code file might have four lines).

|laby| Laby and |bugy| Bugy
