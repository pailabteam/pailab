REM del coverage.xml
coverage run ./tests/repo_test.py
coverage run -a ./tests/repo_disk_storage_test.py
coverage run -a ./tests/SQLiteJobRunner_test.py
coverage html