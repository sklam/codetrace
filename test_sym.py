from __future__ import absolute_import, division, print_function

import dis

from codetrace import symeval, bytecode

#
# def foo(a):
#     c = 0
#     for i in range(a):
#         c += i
#     return c
#
# def foo(a):
#     c = 0
#     for i in range(a):
#         d = 0
#         for j in range(a):
#             d += j
#         c += d
#     return c


def foo(a, b):
    c = a and b + 1 or b
    for i in range(c):
        c += 1
    return c

dis.dis(foo)


instlist = list(bytecode.disassemble(foo))

tracegraph = symeval.symbolic_evaluate(instlist)
tracegraph.dump()
tracegraph.simplify()

tracegraph.graphviz()
tracegraph.verify()
