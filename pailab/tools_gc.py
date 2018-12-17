import os
from google.cloud import storage
from pailab.numpy_handler_hdf import NumpyHDFStorage


def _get_all_files(directory):
    """Returns set of all files in directory

    Returns:
        set: set with all filenames (including relative paths)
    """
    f = set()
    for path, subdirs, files in os.walk(directory):
        for name in files:
            p = path + '/' + name  # os.path.join(directory, name)
            #path.replace(directory, "") + name
            f.add(p.replace(directory, ''))
    return f


def get_numpy_data(numpy_hdf_store: NumpyHDFStorage, gcs_bucket_name: str, update_new_only: bool =True):
    """Get NumpyData for a given NumpyHDFStorage from gcs.

    It only updates new files or files that have a different md5 checksum.

    Args:
        numpy_hdf_store (NumpyHDFStorage): [description]
        gcs_bucket_name (str): [description]
        update_new_only (bool): if True, only new files are loaded from gcs, otherwise also the files with diferent md5 checksum are loaded
    """
    files_in_store = _get_all_files(numpy_hdf_store.main_dir)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(gcs_bucket_name)

    blobs = bucket.list_blobs()

    print(files_in_store)
    for b in blobs:
        print('blob: ' + b.name)
