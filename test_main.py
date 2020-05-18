import unittest
from models import execution

class MainTest(unittest.TestCase):
    def test_process_video(self):
        self.assertEqual(execution.process_video(), True)



def run_testing():
    TestSuite = unittest.TestSuite()
    TestSuite.addTest(unittest.makeSuite(MainTest))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(TestSuite)