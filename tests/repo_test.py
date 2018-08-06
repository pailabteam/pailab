import unittest
from repo.repo_objects import RepoInfoKey # pylint: disable=E0401
from repo.repo_objects import repo_object_init # pylint: disable=E0401
import repo.repo_objects as repo_objects

class TestClass:
    @repo_object_init()
    def __init__(self, a ,b):
        self.a = a
        self._b = b

    def a_plus_b(self):
        return self.a + self._b

    
class RepoInfoTest(unittest.TestCase):
    
    def test_repo_info(self):
        name = 'dummy'
        version = '1.0.2'
        repo_info = repo_objects.RepoInfo(**{ 'version': version, 'NAME': name})
        self.assertEqual(repo_info.name, 'dummy')
        self.assertEqual(repo_info['name'], 'dummy')
        self.assertEqual(repo_info['NAME'], 'dummy')
        self.assertEqual(repo_info[RepoInfoKey.NAME], 'dummy')
        repo_info.set_fields({RepoInfoKey.BIG_OBJECTS: {'dummy': 'dummy2'}})
        self.assertEqual(repo_info.name, 'dummy')
        self.assertEqual(repo_info.big_objects['dummy'], 'dummy2')
        self.assertEqual(repo_info.big_objects['dummy'], 'dummy2')
        repo_info[RepoInfoKey.NAME] = 'dummy2'
        self.assertEqual(repo_info.name, 'dummy2')
        repo_info['NAME'] = 'dummy2'
        self.assertEqual(repo_info['NAME'], 'dummy2')
        repo_info['name'] = 'dummy2'
        self.assertEqual(repo_info['NAME'], 'dummy2')
        self.assertEqual(repo_info['dummy'], None)
        repo_info_dict = repo_info.get_dictionary()
        self.assertEqual(repo_info_dict['name'], 'dummy2')
        # now test if dictionary contains only values for RepoInfoKeys
        for k in RepoInfoKey:
            self.assertEqual(repo_info_dict[k.value], repo_info[k])

class RepoObjectTest(unittest.TestCase):

    def test_repo_object_helper(self):
        orig = TestClass(5, 3)
        self.assertEqual(orig.a_plus_b(), 8)
        orig = TestClass(5, 3, repo_info = {'name': 'test'})
        tmp = repo_objects.create_repo_obj_dict(orig)
        new = repo_objects.create_repo_obj(tmp)
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())
        # test if different constructor call work correctly after applying decorator
        orig = TestClass(5, b=3, repo_info = {'name': 'test'})
        tmp = repo_objects.create_repo_obj_dict(orig)
        new = repo_objects.create_repo_obj(tmp)
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())

        orig = TestClass(**{'a' : 5, 'b' : 3}, repo_info = {'name': 'test'})
        tmp = repo_objects.create_repo_obj_dict(orig)
        new = repo_objects.create_repo_obj(tmp)
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())

    def test_repo_object(self):
        # initialize repo object from test class
        version = 5
        obj = TestClass(5, 3, repo_info = {'name' : 'dummy', 'version': version})
        self.assertEqual(obj.repo_info.name, 'dummy') # pylint: disable=E1101
        self.assertEqual(obj.repo_info.version, version) # pylint: disable=E1101
        repo_dict = obj.to_dict() # pylint: disable=E1101
        self.assertEqual(repo_dict['a'], obj.a)
        self.assertEqual(repo_dict['_b'], obj._b)
        obj2 = TestClass(**repo_dict, repo_info = {'name' : 'dummy', 'version': version}, _init_from_dict = True)
        self.assertEqual(obj2.a, obj.a)
        self.assertEqual(obj2.a_plus_b(), obj.a_plus_b())
       

if __name__ == '__main__':
    unittest.main()
