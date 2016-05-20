import sys
import os.path
import tempfile
import inspect
import unittest
import importlib
from contextlib import contextmanager

from . import samples
from .sample_decompile_tests import DecompileTestCase
from codetrace import symeval, bytecode, cfanalyze, decompiler


@contextmanager
def patch_sys_path(dirpath):
    old = sys.path
    sys.path = [dirpath] + sys.path.copy()
    yield
    sys.path = old


one_args = [0], [1], [2], [10], [11], [12], [21], [29]
one_args = [[10]]


class TestDecompile(DecompileTestCase, unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
