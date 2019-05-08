REM del coverage.xml
coverage run --source=pailab --omit=*/tools_gc.py,*/analysis/*,*/externals/*,*/tools/* ./tests/repo_test.py 
coverage run --source=pailab -a --omit=*/tools_gc.py,*/analysis/*,*/externals/*,*/tools/* ./tests/repo_disk_storage_test.py
coverage run --source=pailab -a --omit=*/tools_gc.py,*/analysis/*,*/externals/*,*/tools/* ./tests/SQLiteJobRunner_test.py
coverage run --source=pailab -a --omit=*/tools_gc.py,*/analysis/*,*/externals/*,*/tools/* ./tests/repo_git_handler_test.py
coverage run --source=pailab -a --omit=*/tools_gc.py,*/analysis/*,*/externals/*,*/tools/* ./tests/tutorial_test.py

coverage html 
REM --omit=*/tools_gc.py,*/analysis/*,*/externals/*,*/tools/*