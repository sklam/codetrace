from __future__ import absolute_import, division, print_function

import dis


def disassemble(obj):
    if hasattr(obj, '__code__'):
        return disassemble_code(obj.__code__)
    else:
        return disassemble_code(obj)


def disassemble_code(co):
    """
    A generator to disassemble a code object; yielding a dictionary per
    bytecode instruction.
    The dictionaries must contain the following fields.
        - op:      [str]   opcode name
        - lineno:  [int]   source line
    The following are optional fields:
        - arg:     [int] raw argument value.
        - label:   [bool] whether it is a label.  default to False.
        - const:          value of the constant
        - name:    [str]  global name
        - varname: [str]  local name
        - compare: [str]  comparison operator
        - free:    [str]  free variable name
    """
    code = co.co_code
    labels = dis.findlabels(code)
    linestarts = dict(dis.findlinestarts(co))

    i = 0
    lineno = 0
    while i < len(code):
        lineno = linestarts.get(i, lineno)
        res = {'pc': i, 'lineno': lineno}
        op = code[i]
        i += 1
        extarg = 0
        if op == dis.EXTENDED_ARG:
            op = code[i]
            arg = code[i:i + 2]
            extarg = arg[0] << 16 | arg[1] << 24
            i += 2

        res['op'] = dis.opname[op]
        if i in labels:
            res['label'] = True
        if op >= dis.HAVE_ARGUMENT:
            arg = code[i:i + 2]
            oparg = arg[0] | arg[1] << 8 | extarg
            res['arg'] = oparg
            i += 2
            if op in dis.hasconst:
                res['const'] = co.co_consts[oparg]
            elif op in dis.hasname:
                res['name'] = co.co_names[oparg]
            elif op in dis.hasjrel:
                res['to'] = i + oparg
            elif op in dis.hasjabs:
                res['to'] = oparg
            elif op in dis.haslocal:
                res['varname'] = co.co_varnames[oparg]
            elif op in dis.hascompare:
                res['compare'] = dis.cmp_op[oparg]
            elif op in dis.hasfree:
                ncell = len(dis.co_cellvars)
                res['free'] = (co.co_cellvars[oparg]
                               if oparg < ncell
                               else co.co_freevars[oparg - ncell])
            # otherwise, it is operating base on stack value only

        res['next'] = i
        yield res
