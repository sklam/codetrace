import unittest

from .sample_decompile_tests import DecompileTestCase, DecompileBase
from codetrace.passes.typeinfer import type_inference


class RewriteBase(object):

    def rewrite(self, pyfunc, tracegraph, typeinfos):
        self.original_tracegraph = tracegraph
        rewritten = type_inference(tracegraph, typeinfos)
        rewritten.simplify()
        rewritten.verify()
        rewritten.graphviz()
        self.rewritten_tracegraph = rewritten
        return rewritten

#
# class TestConstProp(RewriteBase, DecompileTestCase, unittest.TestCase):
#     pass


class TestTypeInfer(RewriteBase, DecompileBase, unittest.TestCase):

    def test_collapsed_ifelse1(self):
        def fold_ifelse(a):
            if isinstance(a, int):
                c = a + 1
            else:
                c = a + 2
            return c + 3

        typeinfos = {'a': int}
        self.check_decompile(fold_ifelse, [(1,)], typeinfos=typeinfos)
        # there should be a reduction of states
        # self.assertLess(len(self.rewritten_tracegraph),
        #                 len(self.original_tracegraph))
        # there is exactly one states
        # self.assertEqual(len(self.rewritten_tracegraph), 1)


if __name__ == '__main__':
    unittest.main()
