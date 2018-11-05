import unittest
import os
import subprocess

import nbformat

main_path = 'examples\\'


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    dirname, nb_name = os.path.split(main_path + path)
    curr_dir = os.getcwd()
    os.chdir(dirname)
    args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=600",
            "--output", 'tmp.ipynb', nb_name]  # fout.name, path]
    FNULL = open(os.devnull, 'w')
    subprocess.check_call(args, stdout=FNULL, stderr=subprocess.STDOUT)
    f = open('tmp.ipynb', 'r')
    nb = nbformat.read(f, nbformat.current_nbformat)
    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]
    f.close()
    os.remove('tmp.ipynb')
    os.chdir(curr_dir)
    return nb, errors


def test_ipynb(name):
    nb, errors = _notebook_run(name)
    return errors == []


class Test_notebooks(unittest.TestCase):

    def test_boston_housing(self):
        self.assertTrue(test_ipynb(
            'boston_housing\\boston_housing.ipynb'))


if __name__ == '__main__':
    unittest.main()
