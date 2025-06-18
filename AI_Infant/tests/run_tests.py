# full, runnable code here
import unittest
import sys
import os

def run_all_tests():
    """
    Discovers and runs all unit tests in the 'tests' directory.
    """
    # Adjust path to make 'src' importable from tests
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Discover all test files in the 'tests' directory
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='test_*.py')
    
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with a non-zero status code if any tests failed
    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == '__main__':
    run_all_tests()