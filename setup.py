from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


install_requirements = [
    "numpy>=1.13",
    "plotly>=3.5.0",
    "h5py>=2.7",
    "pandas>=0.20",
    "nbformat",
    "deepdiff",
]


# Extra dependencies for externals
sklearn = [
    "sklearn",
]
tensorflow = [
    "tensorflow",
]

all_dependencies = sklearn + tensorflow

setup(name='pailab',
      version='0.0.3',
      description='Pailab the ai and large data platform',
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
      ],
      keywords='platform ml version-control neural-network ml-workbench ai large data',
      url='https://github.com/pailabteam/pailab',
      author='pailabteam',
      author_email='pailab@rivacon.com',
      license='Apache 2.0',
      packages=find_packages(exclude=['doc', 'examples', 'tests']),
      install_requires=install_requirements,
          extras_require={
          'all': all_dependencies,
          'sklearn': sklearn,
          'tensorflow': tensorflow,
          ':python_version>="3.4"': ['futures'],
      },
      include_package_data=True,
      zip_safe=False)
