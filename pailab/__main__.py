"""Main entry point for pailab command line """
import argparse
import sys
import getpass
from pailab import MLRepo
#from pailab import main


def _get_default_config():
    return {'user': 'test_user',
            'workspace': 'c:/dummy',
            'repo_store':
            {
                'type': 'git_handler',
                'config': {
                    'folder': 'c:/dummy',
                    'file_format': 'json',
                    'remote': 'https://github.com/pailabteam/scott_chesney.git'
                }
            },
            'numpy_store':
            {
                'type': 'hdf_remote_handler',
                'config': {
                    'folder': 'c:/dummy',
                    'remote_store': {
                        'type': 'gcs',
                        'config': {
                            'bucket': 'dummy_bucket',
                            'project': 'dummy_project'
                        }
                    },
                    'sync_get': False,
                    'sync_add': True
                }
            },
            'job_runner':
            {
                'type': 'simple',
                'config': {
                    'throw_job_error': True
                }
            }
            }


def _set_numpy_store_config(argument, config):
    """[summary]

    Examples:

        hdf_remote_handler@gcs:project:bucket
    Args:
        args ([type]): [description]
    """
    result = {}
    tmp = argument.split('@')
    store_type = tmp[0]
    config['numpy_store']['type'] = store_type
    if store_type == 'hdf_remote_handler':
        if len(tmp) < 2:
            raise Excetion(
                'Please specify a remote, e.g. hdf_remote_handler@remote_settings.')
        tmp = tmp[1].split(':')
        if tmp[0] == 'gcs':
            if len(tmp) < 3:
                raise Exception(
                    'Please specify project and bucket for gcs, e.g. hdf_remote_handler@gcs:project:bucket.')
            config['numpy_store']['config']['remote_store']['type'] = 'gcs'
            config['numpy_store']['config']['remote_store']['config']['project'] = tmp[1]
            config['numpy_store']['config']['remote_store']['config']['bucket'] = tmp[2]
            return
        raise Exception('Unknown remote type ' + tmp[0])
    raise Exception('Unknown store type ' + store_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('command', help='one of the command(s) create|init')
    parser.add_argument('workspace', help='directory for the workspace')
    parser.add_argument(
        'obj_handler', type=str, help='definition of store for the repo objects')
    parser.add_argument(
        'bigobj_handler', type=str, help='definition of store for the big object parts of the repo objects')
    parser.add_argument('-u', '--user', type=str,
                        help='username used internally, if not specified, the operating system user is used')
    args = parser.parse_args()
    default_config = _get_default_config()
    default_config['workspace'] = args.workspace
    if args.user:
        default_config['user'] = args.user
    else:
        default_config['user'] = getpass.getuser()
    default_config['numpy_store']['config']['folder'] = args.workspace + '/hdf'
    default_config['repo_store']['config']['folder'] = args.workspace + '/objects'
    _set_numpy_store_config(args.bigobj_handler, default_config)
    tmp = args.obj_handler.split('@')
    default_config['repo_store']['type'] = tmp[0]
    if tmp[0] == 'git_handler':
        if len(tmp) > 1:
            default_config['repo_store']['config']['remote'] = tmp[1]

    #print('Creating repo with config: ' + str(default_config))
    ml_repo = MLRepo(config=default_config, save_config=True)
