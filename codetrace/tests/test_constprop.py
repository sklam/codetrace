import unittest

from .sample_decompile_tests import DecompileTestCase, DecompileBase
from codetrace import partialeval


class RewriteBase(object):

    def rewrite(self, pyfunc, tracegraph):
        self.original_tracegraph = tracegraph
        rewritten = partialeval.partial_evaluate(tracegraph)
        rewritten.simplify()
        rewritten.verify()
        self.rewritten_tracegraph = rewritten
        return rewritten


class TestConstProp(RewriteBase, DecompileTestCase, unittest.TestCase):
    pass


class TestConstPropSpecial(RewriteBase, DecompileBase, unittest.TestCase):
    view_region_tree = False

    def test_collapsed_ifelse1(self):
        def fold_ifelse():
            a = 1
            if a > 0:
                a -= 2
            else:
                a += 3
            return a

        self.check_decompile(fold_ifelse, [()])
        self.assertLess(len(self.rewritten_tracegraph),
                        len(self.original_tracegraph))
        self.assertEqual(len(self.rewritten_tracegraph), 1)


if __name__ == '__main__':
    unittest.main()
