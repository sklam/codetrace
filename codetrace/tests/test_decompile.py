import unittest

from .sample_decompile_tests import DecompileTestCase


class TestDecompile(DecompileTestCase, unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
