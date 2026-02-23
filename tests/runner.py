import unittest


def run_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)


if __name__ == "__main__":
    run_tests()
