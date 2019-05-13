import os
from google.cloud import storage
from pailab.numpy_handler_hdf import NumpyHDFStorage


def get_numpy_data(numpy_hdf_store: NumpyHDFStorage, gcs_bucket_name: str, update_new_only: bool = True):
    """ Get NumpyData for a given NumpyHDFStorage from gcs.

    It only updates new files or files that have a different md5 checksum.

    Arguments:
        numpy_hdf_store {NumpyHDFStorage} -- the numpy hdf storage
        gcs_bucket_name {str} -- [description]

    Keyword Arguments:
        update_new_only {bool} -- if True, only new files are loaded from gcs, otherwise also the files with diferent md5 checksum are loaded (default: {True})
    """

    files_in_store = _get_all_files(numpy_hdf_store.main_dir)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(gcs_bucket_name)

    blobs = bucket.list_blobs()

    print(files_in_store)
    for b in blobs:
        print('blob: ' + b.name)


class RemoteGCS:
    def __init__(self, bucket='', project=None, credentials=None):
        self._storage_client = storage.Client(
            project=project, credentials=credentials)
        self._bucket = self._storage_client.get_bucket(bucket)
        self._bucket_name = bucket
        self._project = project

    def _get_bucket(self):
        return self._bucket

    def _remote_file_list(self):
        return self._get_bucket().list_blobs()

    def _download_file(self, local_filename, remote_filename):
        bucket = self._get_bucket()
        remote_filename = remote_filename.replace('\\', '/')
        blob = bucket.blob(remote_filename)
        blob.download_to_filename(local_filename)

    def _upload_file(self,  local_filename, remote_filename):
        bucket = self._get_bucket()
        remote_filename = remote_filename.replace('\\', '/')
        blob = bucket.blob(remote_filename)
        blob.upload_from_filename(local_filename)

    def file_exists(self, filename):
        filename = filename.replace('\\', '/')
        return storage.Blob(bucket=self._bucket, name=filename).exists(self._storage_client)
