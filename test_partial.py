from __future__ import absolute_import, division, print_function

import dis

from codetrace import symeval, bytecode, partialeval


def foo(a, b):
    d = True
    if d:
        c = 1
    else:
        c = 2
    for i in range(c):
        c += i
    return c


def foo(a, b):
    d = True
    if d:
        c = 1
    else:
        c = 2

    for i in range(c):
        c += 1

    if not d:
        c += 10
    else:
        c += 20

    return c


dis.dis(foo)
instlist = list(bytecode.disassemble(foo))
model = symeval.symbolic_evaluate(instlist)
model.simplify()
model.graphviz()
model = partialeval.partial_evaluate(model)

model.graphviz(filename='presimplify.dot')
model.simplify()
model.graphviz(filename='postsimplify.dot')
# model.dump()
model.verify()
