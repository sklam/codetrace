import unittest

from .sample_decompile_tests import DecompileTestCase, DecompileBase
from codetrace.passes.constprop import constant_propagation


class RewriteBase(object):

    def rewrite(self, pyfunc, tracegraph):
        self.original_tracegraph = tracegraph
        rewritten = constant_propagation(tracegraph)
        rewritten.simplify()
        rewritten.verify()
        self.rewritten_tracegraph = rewritten
        return rewritten


class TestConstProp(RewriteBase, DecompileTestCase, unittest.TestCase):
    pass


class TestConstPropSpecial(RewriteBase, DecompileBase, unittest.TestCase):

    def test_collapsed_ifelse1(self):
        def fold_ifelse():
            a = 1
            if a > 0:
                a -= 2
            else:
                a += 3
            return a

        self.check_decompile(fold_ifelse, [()])
        # there should be a reduction of states
        self.assertLess(len(self.rewritten_tracegraph),
                        len(self.original_tracegraph))
        # there is exactly one states
        self.assertEqual(len(self.rewritten_tracegraph), 1)

    def test_collapsed_loop(self):
        def fold_loop(a, b):
            c = 3
            for i in range(b):
                if c == 3:
                    return a

        self.check_decompile(fold_loop, [(100, 1), (200, 0)])
        # there should be a reduction of states
        self.assertLess(len(self.rewritten_tracegraph),
                        len(self.original_tracegraph))
        # there is exactly three states
        self.assertEqual(len(self.rewritten_tracegraph), 3)


if __name__ == '__main__':
    unittest.main()
