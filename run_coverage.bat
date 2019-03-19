#del coverage.xml
coverage run --source=pailab ./tests/repo_test.py
coverage run --source=pailab -a ./tests/repo_disk_storage_test.py
coverage run --source=pailab -a ./tests/SQLiteJobRunner_test.py
coverage run --source=pailab -a ./tests/repo_git_handler_test.py
coverage html