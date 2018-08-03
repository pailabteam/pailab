import unittest
from repo.repo_objects import RepoInfoKey # pylint: disable=E0401
from repo.repo_objects import repo_object_init # pylint: disable=E0401
import repo.repo as repo
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
        self.assertEqual(repo_info[RepoInfoKey.Name], 'dummy')
        self.assertEqual(repo_info[RepoInfoKey.Name.value], 'dummy')

        repo_info.set_fields({RepoInfoKey.BIG_OBJECTS: {'dummy': 'dummy2'}})
        self.assertEqual(repo_info.name, 'dummy')
        self.assertEqual(repo_info.big_objects['dummy'], 'dummy2')

    
class RepoObjectTest(unittest.TestCase):

    def test_repo_object_helper(self):
        orig = TestClass(5, 3)
        tmp = repo_objects._create_repo_dictionary(orig)
        new = repo_objects._create_object_from_repo_object(tmp)
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())
        # test if different constructor call work correctly after applying decorator
        orig = TestClass(5, b=3)
        tmp = repo_objects._create_repo_dictionary(orig)
        new = repo_objects._create_object_from_repo_object(tmp)
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())

        orig = TestClass(a=5, b=3)
        tmp = repo_objects._create_repo_dictionary(orig)
        new = repo_objects._create_object_from_repo_object(tmp)
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())

        orig = TestClass(**{'a' : 5, 'b' : 3})
        tmp = repo_objects._create_repo_dictionary(orig)
        new = repo_objects._create_object_from_repo_object(tmp)
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())

    def test_repo_object(self):
        # initialize repo object from test class
        orig = TestClass(5, 3)
        version = '1.0.1'
        obj = repo_objects.RepoObject(orig, repo_info = {'name' : 'dummy', 'version': version})
        self.assertEqual(obj.repo_info.name, 'dummy')
        self.assertEqual(obj.repo_info.version, version)
       
        # check if class creation works
        #new = obj.get_class()
        #self.assertEqual(orig.a_plus_b(), new.a_plus_b())
       

if __name__ == '__main__':
    unittest.main()
