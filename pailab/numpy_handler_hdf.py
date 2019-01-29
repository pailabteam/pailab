import h5py
import os
import pathlib
import logging
from pailab.repo_store import NumpyStore
logger = logging.getLogger(__name__)


def trace(aFunc):
    """Trace entry, exit and exceptions."""
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

    def __init__(self, main_dir):
        self.main_dir = main_dir
        if not os.path.exists(self.main_dir):
            os.makedirs(self.main_dir)

    @staticmethod
    @trace
    def _save(data_grp, ref_grp, numpy_dict):
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
                                k, data=tmp.regionref[0:v.shape[0], 0:v.shape[1], 0:v.shape[1]])
                        else:
                            raise NotImplementedException(
                                'Not implemenet for dim>3.')

    @trace
    def add(self, name, version, numpy_dict):
        """ Add numpy data from an object to the storage.

        :param name: Name (as string) of object
        :param version: object version
        :param numpy_dict: numpy dictionary

        """
        # dir_name = os.path.dirname(self.main_dir + '/' + name )
        tmp = pathlib.Path(self.main_dir + '/' + name + 'hdf')
        save_dir = tmp.parent
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with h5py.File(self.main_dir + '/' + name + '.hdf5', 'a') as f:
            grp_name = '/data/' + version + '/'
            logging.debug('Saving data ' + name +
                          ' in hdf5 to group ' + grp_name)
            grp = f.create_group(grp_name)
            ref_grp = f.create_group('/ref/' + str(version) + '/')
            NumpyHDFStorage._save(grp, ref_grp, numpy_dict)

    @trace
    def append(self, name, version_old, version_new, numpy_dict):
        with h5py.File(self.main_dir + '/' + name + '.hdf5', 'a') as f:
            logging.debug('Appending data ' + name +
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

    @trace
    def get(self, name, version, from_index=0, to_index=None):
        # \todo auch hier muss die referenz einschl. Namen verwendet werden
        with h5py.File(self.main_dir + '/' + name + '.hdf5', 'a') as f:
            grp_name = '/data/' + str(version) + '/'
            ref_grp = '/ref/' + str(version) + '/'
            logging.debug('Reading object ' + name +
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
        result = False
        try:
            with h5py.File(self.main_dir + '/' + name + '.hdf5', 'a') as f:
                grp_name = '/data/' + str(version) + '/'
                result = grp_name in f
        except:
            pass
        return result
