del coverage.xml
REM coverage run ./tests/repo_test.py
run_coverage.bat
codecov --token 4662074f-5b97-431e-b671-c2a2454689bc
coverage html
pause