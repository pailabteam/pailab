|Docs|_ |Travis|_

.. |Travis| image:: https://travis-ci.org/pailabteam/pailab.svg?branch=feature%2Ftravis
.. _Travis: https://travis-ci.org/pailabteam/pailab

.. |Codecov| image:: https://codecov.io/gh/pailabteam/pailab/branch/feature%2Ftravis/graph/badge.svg
.. _Codecov:  :target: https://codecov.io/gh/pailabteam/pailab

.. |Docs| image:: https://readthedocs.org/projects/pailab/badge/?version=latest
.. _Docs: https://pailab.readthedocs.io/en/latest/?badge=latest

pailab
==============
Pailab is an integrated machine learning environment to version, analyze and automatize the machine learning model building processes and deployments.
It keeps track of changes in your machine learning pipeline (code, data, parameters) similar to classical 
version control systems considering special characteristics of ai model building processes. 

Al objects added to the repositor are split into a part containing big data and a part with the remaining data handling them separately with different technqiues. For example
one may use git to administrate the remaining data part. It also adds information to each object such as
the version, the author, date etc. and each object is labeled with a category defining the role in the ml process. 