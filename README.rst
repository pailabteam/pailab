.. image:: https://codecov.io/gh/favdo/ml_repo/branch/develop/graph/badge.svg?token=hlafckvwJw
  :target: https://codecov.io/gh/favdo/ml_repo


pailab
==============
Pailab is an integrated machine learning environment to version, analyze and automatize the machine learning model building processes and deployments.
It keeps track of changes in your machine learning pipeline (code, data, parameters) similar to classical 
version control systems considering special characteristics of ai model building processes. 

Al objects added to the repositor are split into a part containing big data and a part with the remaining data handling them separately with different technqiues. For example
one may use git to administrate the remaining data part. It also adds information to each object such as
the version, the author, date etc. and each object is labeled with a category defining the role in the ml process. 