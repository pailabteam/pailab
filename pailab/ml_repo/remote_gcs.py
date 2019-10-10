import os
import logging
from google.cloud import storage
logger = logging.getLogger(__name__)


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
        return [x.name for x in self._get_bucket().list_blobs()]

    def _download_file(self, local_filename, remote_filename):
        bucket = self._get_bucket()
        remote_filename = remote_filename.replace('\\', '/')
        blob = bucket.blob(remote_filename)
        # if not os.path.exists(local_filename):
        #    os.path.ma
        path_to_local_file = os.path.dirname(local_filename)
        if not os.path.exists(path_to_local_file):
            os.makedirs(path_to_local_file)
        with open(local_filename, 'wb') as file_obj:
            logger.debug('Start downloading ' +
                         remote_filename + ' to ' + local_filename)
            blob.download_to_file(file_obj)
            logger.debug('Finished downloading ' + remote_filename)

    def _upload_file(self,  local_filename, remote_filename):
        bucket = self._get_bucket()
        remote_filename = remote_filename.replace('\\', '/')
        blob = bucket.blob(remote_filename)
        with open(local_filename, 'rb') as file_obj:
            logger.debug('Start uploading file ' +
                         local_filename + ' to ' + remote_filename)
            blob.upload_from_file(file_obj)
            logger.debug('Start uploading file ' + local_filename)

    def file_exists(self, filename):
        filename = filename.replace('\\', '/')
        return storage.Blob(bucket=self._bucket, name=filename).exists(self._storage_client)

    def _delete_file(self,  filename):
        bucket = self._get_bucket()
        filename = filename.replace('\\', '/')
        blob = bucket.blob(filename)
        blob.delete()
