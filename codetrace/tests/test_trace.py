from __future__ import absolute_import, print_function
from codetrace import tracing, emulator
from .support import unittest


class TestTrace(unittest.TestCase):
    def check_trace(self, fn, cases, errcases=(), show_dot=False):
        traces = tracing.trace(fn)
        if show_dot:
            traces.view_dot_graph()
        emu = emulator.Emulator(traces)
        print(emu.source_code)
        # test cases that don't raise
        for num, args in enumerate(cases):
            with self.subTest(num=num, args=args):
                self.assertEqual(emu(*args), fn(*args))
        # test cases that raise
        for num, (exctype, args) in enumerate(errcases, start=num):
            with self.subTest(num=num, args=args):
                with self.assertRaises(exctype) as expected:
                    fn(*args)
                with self.assertRaises(exctype) as actual:
                    emu(*args)
                self.assertEqual(expected.exception.args, actual.exception.args)

    def test_add(self):
        def foo(a, b):
            return a + b
        cases = [(1, 2), (4, 3), (-4, 3)]
        self.check_trace(foo, cases)

    def test_or(self):
        def foo(a, b):
            return a or b
        cases = [(1, 2), (0, 3), (-1, 0)]
        self.check_trace(foo, cases)

    def test_if_else_max(self):
        def foo(a, b):
            if a < b:
                return b
            else:
                return a
        cases = [(1, 2), (4, 3), (3, 3)]
        self.check_trace(foo, cases)

    def test_if_else_return_after(self):
        def foo(a, b):
            res = None
            if a < b:
                res = a
            elif a > b:
                res = b
            return res

        cases = [(1, 2), (4, 3), (3, 3)]
        self.check_trace(foo, cases)

    def test_for_range_loop(self):
        def foo(a, b):
            c = 0
            for i in range(a, b):
                c = c + i
            return c
        cases = [(0, 10), (5, 5), (-5, 5)]
        self.check_trace(foo, cases)

    def test_for_range_nested(self):
        def foo(a, b):
            c = 0
            for i in range(a):
                for j in range(b):
                    c = (i + 1) * (j + 1)
            return c
        cases = [(5, 7), (1, 2), (4, 0)]
        self.check_trace(foo, cases)

    def test_try_except(self):
        def foo(a):
            c = 0
            try:
                c = a.notreally
            except:
                c = 2
            return c

        class Dummy(object):
            notreally = 1

        cases = [(0,), (Dummy,)]
        self.check_trace(foo, cases)

    def test_try_except_else(self):
        def foo(a):
            c = 0
            d = 1
            try:
                d = a.notreally
            except:
                c = 2
            else:
                c = 3 + d
            return c

        class Dummy(object):
            notreally = 10

        cases = [(0,), (Dummy,)]
        self.check_trace(foo, cases)

    def test_try_except_matched(self):
        def foo(a):
            c = 0
            try:
                c = a.notreally
            except AttributeError:
                c = 2
            return c

        class Dummy(object):
            notreally = 1

        class Funky(object):
            @property
            def notreally(self):
                raise ValueError("over here")

        cases = [(0,), (Dummy,)]
        errcases = [(ValueError, (Funky(),))]
        self.check_trace(foo, cases, errcases)

    def test_try_except_neseted(self):
        def foo(a):
            c = 0
            d = 0
            try:
                try:
                    x = a.one()
                except ValueError:
                    c = a.two()
                else:
                    c = 2 * x

                y = a.three()
            except TypeError:
                d = 3
            else:
                d = 4 * y

            return d * 100 + c

        class Case1(object):
            def one(self):
                return 7

            def three(self):
                return 8

        class Case2(object):
            def one(self):
                return ValueError("from one")

            def two(self):
                raise 6

            def three(self):
                return 9

        class Case3(object):
            def one(self):
                return ValueError("from one")

            def two(self):
                raise TypeError("from two")

            def three(self):
                return 8

        class Case4(object):
            def one(self):
                return ValueError("from one")

            def two(self):
                raise 3

            def three(self):
                raise TypeError("from three")

        cases = [(Case1(),),
                 (Case2(),),
                 (Case3(),),
                 (Case4(),)]
        errcases = [(AttributeError, (None,))]
        self.check_trace(foo, cases, errcases)


if __name__ == '__main__':
    unittest.main()
