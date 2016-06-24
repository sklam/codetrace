import sys
import os.path
import tempfile
import inspect
import importlib
from contextlib import contextmanager

from . import samples
from codetrace import symeval, bytecode, cfanalyze, decompiler


@contextmanager
def patch_sys_path(dirpath):
    old = sys.path
    sys.path = [dirpath] + sys.path.copy()
    yield
    sys.path = old


one_args = [0], [1], [2], [10], [11], [12], [21], [29]
one_args = [[10]]


class DecompileBase(object):

    view_region_tree = False

    def decompile(self, pyfunc, **kwargs):
        instlist = list(bytecode.disassemble(pyfunc))

        tracegraph = symeval.symbolic_evaluate(instlist)
        tracegraph.simplify()
        tracegraph.verify()

        tracegraph = self.rewrite(pyfunc, tracegraph, **kwargs)

        cfa = cfanalyze.CFA(tracegraph)
        sig = inspect.signature(pyfunc)

        # Plot region tree
        if self.view_region_tree:
            cfa.gv_region_tree(tracegraph, filename=pyfunc.__name__ + '.gv')

        fname = pyfunc.__name__
        code = decompiler.decompile(tracegraph, cfa, fname, sig)
        return code

    def rewrite(self, pyfunc, tracegraph):
        return tracegraph

    def dynamic_import(self, module_name, fname):
        invalidate_caches = getattr(importlib, 'invalidate_caches',
                                    None)
        if invalidate_caches:
            invalidate_caches()

        module = importlib.import_module(module_name)
        return getattr(module, fname)

    def write_code_to_file(self, srcfile, code):
        print('writing to', srcfile.name)
        print('import codetrace.decompiled_runtime as __rt__', file=srcfile)
        print(code, file=srcfile)
        srcfile.flush()

    def check_decompile(self, pyfunc, args, **kwargs):
        code = self.decompile(pyfunc, **kwargs)
        fname = pyfunc.__name__
        assert args, 'no test arguments'
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py') as srcfile:
            self.write_code_to_file(srcfile, code)

            dirpath = os.path.dirname(srcfile.name)
            module_name, ext = os.path.splitext(os.path.basename(srcfile.name))
            with patch_sys_path(dirpath):
                defunc = self.dynamic_import(module_name, fname)
                # execute
                for curarg in args:
                    with self.subTest(arg=curarg):
                        print('expected'.center(80, '-'))
                        expect = pyfunc(*curarg)

                        try:
                            print('got'.center(80, '-'))
                            got = defunc(*curarg)
                            self.assertEqual(expect, got)
                        except:

                            # import pudb; pudb.set_trace()
                            got = defunc(*curarg)
                            with open('temp_check_error.py', 'w') as fout:
                                self.write_code_to_file(fout, code)
                            raise


class DecompileTestCase(DecompileBase):

    def test_ifelse1(self):
        self.check_decompile(samples.ifelse1, args=one_args)

    def test_ifelse2(self):
        self.check_decompile(samples.ifelse2, args=one_args)

    def test_loop1(self):
        self.check_decompile(samples.loop1, args=one_args)

    def test_loop2(self):
        self.check_decompile(samples.loop2, args=one_args)

    def test_loop3(self):
        self.check_decompile(samples.loop3, args=one_args)

    def test_loop4(self):
        self.check_decompile(samples.loop4, args=one_args)

    def test_loop5(self):
        self.check_decompile(samples.loop5, args=one_args)
