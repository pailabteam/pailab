# -*- coding: utf-8 -*-
"""Module defining classes to store numpy data in hdf5 files.

This module provides implementations of the :py:class:`pailab.ml_repo.repo_store.NumpyStore` using hdf5 file format.
"""
import h5py
import os
import pathlib
import logging
from pailab.ml_repo.repo_store import NumpyStore
logger = logging.getLogger(__name__)


def trace(aFunc):
    """ Trace entry, exit and exceptions.
    """
    def loggedFunc(*args, **kw):
        logger.info("Entering " + aFunc.__name__)
        try:
            result = aFunc(*args, **kw)
        except Exception as e:
            logger.exception("Error in " + aFunc.__name__ + ':' + str(e))
            raise
        logger.info("Exit " + aFunc.__name__)
        return result
    loggedFunc.__name__ = aFunc.__name__
    loggedFunc.__doc__ = aFunc.__doc__
    return loggedFunc


class NumpyHDFStorage(NumpyStore):
    """ Storage using hdf5 files to store numpy data.
    
    Example:
        Setup storage using folder ``C:\\temp\\data``::

            >>> store = NumpyHDFStorage('C:\\temp\\data')
    Args:
        folder (str): main directory where the files will be stored
        version_files (bool): If True, each version is contained in a separate file, otherwise all versions are in one file.
            If you like to work in a distributed environmnt (e.g. multiple users working in parallel) you should set this parameter to True so that no file merge is necessary.
             (default: {False})
    
    """

    def __init__(self, folder, version_files=False):
        self.main_dir = folder
        self._version_files = version_files
        if not os.path.exists(self.main_dir):
            os.makedirs(self.main_dir)

    def _create_file_name(self, name, version, change_if_not_exist=False):
        """ Function to create the file name for an object 

        Arguments:
            name {str} -- the identifier object
            version {str} -- the version id

        Keyword Arguments:
            change_if_not_exist {bool} -- change the name if it does not exist (default: {False})

        Returns:
            str -- the filename
        """

        if self._version_files:
            filename = name + '_' + version + '.hdf5'
            if change_if_not_exist:
                if not os.path.exists(self.main_dir + '/' + filename):
                    return name + '.hdf5'
            return filename
        else:
            return name + '.hdf5'

    @trace
    def _delete(self, name, version):
        """ Delete an object with a predefined version

        Arguments:
            name {str} -- identifier of the object
            version {str} -- the version id
        """

        pass

    @staticmethod
    @trace
    def _save(data_grp, ref_grp, numpy_dict):
        """ saving the data 

        Arguments:
            data_grp {[type]} -- the data group
            ref_grp {[type]} -- the reference group 
            numpy_dict {numpy dict} -- the numpy dictionary to save

        Raises:
            NotImplementedException -- [description]
        """

        for k, v in numpy_dict.items():
            if v is not None:
                if len(v.shape) == 1:
                    tmp = data_grp.create_dataset(
                        k, data=v, maxshape=(None, ))
                    ref_grp.create_dataset(k, data=tmp.regionref[0:v.shape[0]])
                else:
                    if len(v.shape) == 2:
                        tmp = data_grp.create_dataset(
                            k, data=v, maxshape=(None, v.shape[1]))
                        ref_grp.create_dataset(
                            k, data=tmp.regionref[0:v.shape[0], 0:v.shape[1]])
                    else:
                        if len(v.shape) == 3:
                            tmp = data_grp.create_dataset(
                                k, data=v, maxshape=(None, v.shape[1], v.shape[2]))
                            ref_grp.create_dataset(
                                k, data=tmp.regionref[0:v.shape[0], 0:v.shape[1], 0:v.shape[2]])
                        else:
                            if len(v.shape) == 4:
                                tmp = data_grp.create_dataset(
                                    k, data=v, maxshape=(None, v.shape[1], v.shape[2], v.shape[3]))
                                ref_grp.create_dataset(
                                    k, data=tmp.regionref[0:v.shape[0], 0:v.shape[1], 0:v.shape[2], 0:v.shape[3]])
                            else:                                        
                                raise NotImplementedException(
                                    'Not implemenet for dim>3.')

    @trace
    def add(self, name, version, numpy_dict):
        """ Add numpy data from an object to the storage.

        Arguments:
            name {str} -- the identifier of the object to add
            version {str} -- the object version 
            numpy_dict {numpy dict} -- the numpy dictionary to add
        """

        tmp = pathlib.Path(self.main_dir + '/' + name + 'hdf')
        save_dir = tmp.parent
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with h5py.File(self.main_dir + '/' + self._create_file_name(name, version), 'a') as f:
            grp_name = '/data/' + version + '/'
            logger.debug('Saving data ' + name +
                         ' in hdf5 to group ' + grp_name)
            grp = f.create_group(grp_name)
            ref_grp = f.create_group('/ref/' + str(version) + '/')
            NumpyHDFStorage._save(grp, ref_grp, numpy_dict)

    @trace
    def _append_same_file(self, name, version_old, version_new, numpy_dict):
        """ Append data to the same file

        Arguments:
            name {str} -- the object identifier
            version_old {str} -- the previous object version
            version_new {str} -- the next object version
            numpy_dict {numpy dict} -- the data to add as a numpy dictionary
        """

        with h5py.File(self.main_dir + '/' + self._create_file_name(name, version_new), 'a') as f:
            logger.debug('Appending data ' + name +
                         ' in hdf5 with version ' + str(version_new))

            try:
                ref_grp = f['/ref/' + str(version_new) + '/']
                grp = f['/data/' + str(version_new) + '/']
            except:
                ref_grp = f.create_group('/ref/' + str(version_new) + '/')
                grp = f.create_group('/data/' + str(version_new) + '/')

            grp_previous = f['/data/' + str(version_old) + '/']
            for k, v in numpy_dict.items():
                data = grp_previous[k]
                old_size = len(data)
                new_shape = [x for x in v.shape]
                new_shape[0] += old_size
                new_shape = tuple(new_shape)
                data.resize(new_shape)
                grp[k] = h5py.SoftLink(data.name)
                if len(data.shape) == 1:
                    data[old_size:new_shape[0]] = v
                    ref_grp.create_dataset(
                        k, data=data.regionref[0:new_shape[0]])
                else:
                    if len(data.shape) == 2:
                        data[old_size:new_shape[0], :] = v
                        ref_grp.create_dataset(
                            k, data=data.regionref[0:new_shape[0], :])
                    else:
                        if len(data.shape) == 3:
                            data[old_size:new_shape[0], :, :] = v
                            ref_grp.create_dataset(
                                k, data=data.regionref[0:new_shape[0], :, :])
                        else:
                            if len(data.shape) == 4:
                                data[old_size:new_shape[0], :, :, :] = v
                                ref_grp.create_dataset(
                                    k, data=data.regionref[0:new_shape[0], :, :, :])

    def _append_different_file(self, name, version_old, version_new, numpy_dict):
        """ Append data to a different file

        Arguments:
            name {str} -- the object identifier
            version_old {str} -- the previous object version
            version_new {str} -- the next object version
            numpy_dict {numpy dict} -- the data to add as a numpy dictionary
        """

        # save data to append in separate file
        self.add(name + '_append', version_new, numpy_dict)
        # get shape
        grp_name_old = '/data/' + str(version_old) + '/'
        grp_name_new = '/data/' + str(version_new) + '/'
        old_filename = self._create_file_name(name, version_old)
        new_filename = self._create_file_name(name, version_new)
        tmp_filename = self._create_file_name(name + '_append', version_new)
        with h5py.File(self.main_dir + '/' + old_filename, 'r') as f_old:
            with h5py.File(self.main_dir + '/' + tmp_filename, 'r') as f_tmp:
                with h5py.File(self.main_dir + '/' + new_filename, 'w') as f_new:
                    ref_grp = f_new.create_group(
                        '/ref/' + str(version_new) + '/')
                    grp = f_new.create_group('/data/' + str(version_new) + '/')

                    for k, v in numpy_dict.items():
                        grp_old_k = f_old[grp_name_old][k]
                        grp_new_k = f_tmp[grp_name_new][k]
                        shape = (
                            grp_old_k.shape[0]+grp_new_k.shape[0], ) + grp_old_k.shape[1:]
                        layout = h5py.VirtualLayout(shape=shape)
                        layout[0:grp_old_k.shape[0]
                               ] = h5py.VirtualSource(grp_old_k)
                        layout[grp_old_k.shape[0]:] = h5py.VirtualSource(grp_new_k)
                        tmp = grp.create_virtual_dataset(k, layout)
                        ref_grp.create_dataset(k, data=tmp.regionref[:])
                        #ref_grp.create_virtual_dataset(k, layout)

    @trace
    def append(self, name, version_old, version_new, numpy_dict):
        """ append data to the an existing object

        Arguments:
            name {str} -- the object identifier
            version_old {str} -- the previous object version
            version_new {str} -- the next object version
            numpy_dict {numpy dict} -- the data to add as a numpy dictionary
        """

        if not self._version_files:
            self._append_same_file(name, version_old, version_new, numpy_dict)
        else:
            self._append_different_file(
                name, version_old, version_new, numpy_dict)

    @trace
    def get(self, name, version, from_index=0, to_index=None):
        """ get the numpy object for a name and a version, rows can be used

        Arguments:
            name {str} -- identifier of the object
            version {str} -- version of the object

        Keyword Arguments:
            from_index {int} -- the index from which the data should be taken (default: {0})
            to_index {int or None} -- the index to which the data is returned (None means till the end) (default: {None})

        Raises:
            Exception -- raises an exception if no object with the name exists
            Exception -- raises an exception if no object and with the version exists 

        Returns:
            numpy array -- the numpy object to return
        """

        # \todo auch hier muss die referenz einschl. Namen verwendet werden
        with h5py.File(self.main_dir + '/' + self._create_file_name(name, version, change_if_not_exist=True), 'r') as f:
            grp_name = '/data/' + str(version) + '/'
            ref_grp = '/ref/' + str(version) + '/'
            logger.debug('Reading object ' + name +
                         ' from hdf5, group ' + grp_name)
            grp = f[grp_name]
            ref_g = f[ref_grp]
            result = {}
            for k, v in ref_g.items():
                result[k] = grp[k][v[()]]
        # todo needs improvement, handle indices in reading
        if from_index != 0 or (to_index is not None):
            tmp = {}
            if from_index != 0 and (to_index is not None):
                for k, v in result.items():
                    tmp[k] = v[from_index:to_index, :]
            else:
                if from_index != 0:
                    for k, v in result.items():
                        tmp[k] = v[from_index:-1, :]
                if to_index is not None:
                    for k, v in result.items():
                        tmp[k] = v[0:to_index, :]
            return tmp
        return result

    def object_exists(self, name, version):
        """ checks whether the object exists

        Arguments:
            name {str} -- the identifier of the object
            version {str} -- the version of the object

        Returns:
            bool -- returns true if the object exists
        """

        result = False
        try:
            with h5py.File(self.main_dir + '/' + self._create_file_name(name, version, change_if_not_exist=True), 'a') as f:
                grp_name = '/data/' + str(version) + '/'
                result = grp_name in f
        except:
            pass
        return result


def _get_all_files(directory):
    """ Returns set of all files in directory

    Arguments:
        directory {str} -- the directory

    Returns:
        set -- set with all filenames (including relative paths)
    """

    f = set()
    for path, subdirs, files in os.walk(directory):
        for name in files:
            p = path + '/' + name  # os.path.join(directory, name)
            p = p.replace(directory, '')
            #path.replace(directory, "") + name
            if p[0] == '\\' or p[0] == '/':
                p = p[1:]
            f.add(p)
    return f

import time
from contextlib import contextmanager

@contextmanager
def _lock_dir(main_dir, wait_time, timeout):
    _time = 0
    while _time < timeout:
        try:
            file = open(main_dir + '/.lock', 'x')
            file.close()
            break
        except:
            time.sleep(wait_time)
            _time += wait_time
    yield
    if _time >= timeout:
        raise Exception('Cannot obtain lock due to timeout. Either increase timeout or remove .lock in ' + main_dir + ' to make everything working.')
    if os.path.exists(main_dir + '/.lock'):
        os.remove(main_dir + '/.lock')

def _create_remote(remote_type, **kwargs):
    if remote_type == 'gcs':
        from pailab.ml_repo.remote_gcs import RemoteGCS
        return RemoteGCS(**kwargs)
    raise Exception('Unknown remote type ' + remote_type)


class NumpyHDFRemoteStorage(NumpyHDFStorage):
    """Storage working like NumpyHDFStorage locally but in addition provides synchronization with a remote.

    This storage stores numpy data in hdf5 files in a directory. It works very similar to the :py:class:`NumpyHDFStorage` with the difference
    that it synchronizes the data with a given remote (downloads and uploads the respective files).

    Example:
        This example shows how to setup the storage so that the data is stored in a 
        local directory and it can be synchronized ith googl ecloud storage::

            >>> numpy = NumpyHDFRemoteStorage('C:\\tmp\\data')
            >>> from pailab.ml_repo.remote_gcs import RemoteGCS
            >>> remote = RemoteGCS(bucket='my_data')
            >>> numpy.set_remote(remote)

    Args:
       folder (str): folder where data is stored
       remote_store (obj or dict): object representing a remote storage (e.g. :py:class:`pailab.ml_repo.remote_gcs.RemoteGCS` for the google cloud storage) or dictionary defining the remote params so that it can be created 
       sync_get (bool): If True, tries to download data automatically if it does not exist locally, otherwise it checks only locally
       sync_add (bool): If True, added data will be directly uploaded to the remote
    """

    def __init__(self, folder, remote_store = None,  sync_get = False, sync_add = False):
        super(NumpyHDFRemoteStorage, self).__init__(folder, version_files = True)
        if isinstance(remote_store, dict):
            self._remote_store = _create_remote(remote_store['type'], **remote_store['config'])
        else:
            self._remote_store = remote_store
        # timeout in sec (waiting for another get/pull/push from another process)
        self._timeout = 10
        # time in seconds to wait if another process is currently pushing/pulling/getting
        self._wait_time = 5
        self._sync_get = sync_get
        self._sync_add = sync_add

    def set_remote(self, remote_store):
        self._remote_store = remote_store

    def get(self, name, version, from_index=0, to_index=None):
        if not self._sync_get:
            return super(NumpyHDFRemoteStorage, self).get(name, version, from_index, to_index)

        result = None
        try:
            result = super(NumpyHDFRemoteStorage, self).get(name, version, from_index, to_index)
        except:
            with _lock_dir(self.main_dir, self._wait_time, self._timeout):
                filename = self._create_file_name(name, version, change_if_not_exist=False)
                self._remote_store._download_file(self.main_dir + '/' + filename, filename)
                result = super(NumpyHDFRemoteStorage, self).get(name, version, from_index, to_index)
        return result

    def add(self, name, version, numpy_dict):
        super(NumpyHDFRemoteStorage, self).add(name, version, numpy_dict)
        if self._sync_add:
            filename = self._create_file_name(name, version)
            self._remote_store._upload_file( self.main_dir + '/' + filename, filename)

    def push(self):
        """ Push changes to an external repo.
        """

        with _lock_dir(self.main_dir, self._wait_time, self._timeout):
            remote_files = {x for x in self._remote_store._remote_file_list()}
            local_files = _get_all_files(self.main_dir)
            if '.lock' in local_files:
                local_files.remove('.lock')
            files_to_push = local_files-remote_files
            for f in files_to_push:
                self._remote_store._upload_file(self.main_dir + '/' + f, f)

    def pull(self):
        """ Pull changes from an external repo
        """
        with _lock_dir(self.main_dir, self._wait_time, self._timeout):
            remote_files = self._remote_store._remote_file_list()
            local_files = _get_all_files(self.main_dir)
            local_files.remove('.lock')
            files_to_pull = remote_files - local_files
            for f in files_to_pull:
                self._remote_store._download_file(self.main_dir + '/' + f, f)

