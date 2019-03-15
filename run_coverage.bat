REM del coverage.xml
coverage run --source=pailab/ml_repo ./tests/repo_test.py
REM coverage run --source=pailab -a ./tests/repo_disk_storage_test.py
REM coverage run --source=pailab -a ./tests/SQLiteJobRunner_test.py
coverage html
pause