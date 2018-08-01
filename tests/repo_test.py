import unittest
import repo
from repo import repo_object_init

class TestClass:
    @repo_object_init()
    def __init__(self, a ,b):
        self.a = a
        self._b = b

    def a_plus_b(self):
        return self.a + self._b

class RepoTest(unittest.TestCase):
    
    def test_repo_object(self):
        orig = TestClass(5, 3)
        tmp = repo._create_repo_object(orig)
        new = repo._create_object_from_repo_object(tmp)
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())
    
class RepoObjectTest(unittest.TestCase):

    def test_repo_object(self):
        # initialize repo object from test class
        orig = TestClass(5, 3)
        obj = repo.RepoObject(orig, 'dummy')
        self.assertEqual(obj.id, 'dummy')
        self.assertEqual(obj.version, 0)
        # initialize repo object from existing repo object
        obj2 = repo.RepoObject(obj, 'dummy2')
        self.assertEqual(obj.id, obj2.id)
        self.assertEqual(obj.version, obj2.version)

        # check if class creation works
        new = obj.get_class()
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())
        new = obj2.get_class()
        self.assertEqual(orig.a_plus_b(), new.a_plus_b())


if __name__ == '__main__':
    unittest.main()
