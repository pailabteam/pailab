import repo_objects

class RepoObjectMemoryStorage:
    def __init__(self):
        self._store = {}

    def add(self, category, name, obj):
        """ Add an object of given category to the storage.

        The objects version will be set to the latest version+1.
    
        :param category: A string describing the category.
        :param name: Name (as string) of object
        :param obj: The object added.

        """
        if not category in self._store.keys():
            self._store[category] = {}
        if not name in self._store.keys():
            self._store[category][name] = [obj]
        else:
            self._store[category][name].append(obj)
        obj.repo_info[repo_objects.RepoInfoKey.VERSION] = len(self._store[category][name])-1

    def get(self, category, name, version=-1):
        """
        """
        if not category in self._store.keys():
            raise Exception('No object ' + name + ' in category ' + category)
        if not name in self._store[category].keys():
            raise Exception('No object ' + name + ' in category ' + category)
        tmp = self._store[category][name]
        if version >= len(tmp):
            raise Exception('No valid versionnumber (' + str(version) + ') for object ' + name + ' in category ' + category)
        return self._store[category][name][version]
