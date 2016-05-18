from io import StringIO
from contextlib import contextmanager

from . import ir
from .controlflow import Region


def decompile(tracegraph, cfa, name, signature):
    dec = Decompiler(tracegraph, cfa, name, signature)
    return dec.decompile()


class Decompiler(object):

    def __init__(self, tracegraph, cfa, name, signature):
        self._tracegraph = tracegraph
        self._cfa = cfa
        self._name = name
        self._signature = signature
        # map head of regions
        self._regionmap = {}
        regiontree = self._cfa.region_tree()
        for sr in regiontree.subregions():
            self._regionmap[sr.first] = sr

    def decompile(self):
        self._writer = PyCodeWriter()
        try:
            self._emit_function_decl()
            with self._writer.indent():
                cfg = self._cfa.cfg
                # emit node predicates
                for node in cfg.nodes():
                    self._writer.println(_predicate(node), '= False')

                self._writer.println(_predicate(cfg.entry_point()), '= True')

                # emit body
                self._decompile(self._cfa.region_tree())
            return self._writer.show()
        except:
            # print(self._writer.show())
            raise

    def _emit_function_decl(self):
        w = self._writer
        w.println('def {0}{1}:'.format(self._name, self._signature))

    def _emit_outgoing_stack(self, state, vals):
        w = self._writer
        names = list(map(_variable, state.incoming_stack))
        lhs = ', '.join(names)
        rhs = ', '.join(list(map(_variable, vals))[:len(names)])
        w.println('[{lhs}] = [{rhs}]'.format(lhs=lhs, rhs=rhs))

    def _decompile(self, item):
        if isinstance(item, Region):
            self._region = item  # remeber current region
            self._decompile_region(item)
        else:
            self._decompile_state(item)

    def _decompile_state(self, state):
        w = self._writer
        w.println('#', state)
        w.println(_predicate(state), '= False')

        # decompile instructions (except terminator)
        insts = list(state)
        for inst in insts[:-1]:
            self._decompile_inst(state, inst)

        self._decompile_terminator(state)

    def _decompile_region(self, region):
        cfg = self._cfa.region_local_cfg(self._tracegraph, region)
        loops = cfg.loops()
        w = self._writer

        postdoms = cfg.post_dominators()
        toponodes = list(filter(lambda x: x is not None, cfg.topo_order()))
        # determine nodes that must be at the top level
        toplevelnodes = frozenset(postdoms[cfg.entry_point()])
        assert cfg.entry_point() in toplevelnodes

        # emit nodes
        if loops:
            looppreds = list(map(_predicate, toponodes))
            loopcond = ' or '.join(looppreds)

            w.println('while {cond}:'.format(cond=loopcond))
            with w.indent():
                self._emit_region_body(toponodes, ())

        else:
            self._emit_region_body(toponodes, toplevelnodes)

    def _emit_node(self, node, toplevelnodes):
        w = self._writer
        pred = _predicate(node)
        if node not in toplevelnodes:
            pred = _predicate(node)
            w.println('if {0}:'.format(pred))
            with w.indent():
                self._decompile(node)

        elif node is not None:
            w.println('assert', pred)
            self._decompile(node)

    def _emit_region_body(self, toponodes, toplevelnodes):
        for node in toponodes:
            self._emit_node(node, toplevelnodes)

    def _decompile_terminator(self, state):
        w = self._writer
        term = state.terminator

        if isinstance(term, ir.JumpIf):
            def getstate(target):
                return self._tracegraph[state, target]
            then_state = getstate(term.then)
            else_state = getstate(term.orelse)
            cond = _variable(term.pred)
            w.println('if {cond}:'.format(cond=cond))
            with w.indent():
                w.println(_predicate(then_state), '= True')
                self._emit_outgoing_stack(then_state,
                                          term.outgoing_stack(term.then))
            w.println('else:')
            with w.indent():
                w.println(_predicate(else_state), '= True')
                self._emit_outgoing_stack(else_state,
                                          term.outgoing_stack(term.orelse))

        elif isinstance(term, ir.Jump):
            target_state = self._tracegraph[state, term.target]
            w.println(_predicate(target_state), '= True')
            self._emit_outgoing_stack(target_state,
                                      term.outgoing_stack(term.target))

        elif isinstance(term, ir.Ret):
            w.println('return {0}'.format(_variable(term.value)))

        else:
            raise NotImplementedError(type(term))

    def _decompile_inst(self, state, inst):
        w = self._writer
        if isinstance(inst, ir.Use):
            use = inst
            inst = inst.value

            def set_use(val):
                w.println(_variable(use), '=', val)

        if isinstance(inst, ir.Meta):
            # w.println('#', str(inst))
            pass

        elif isinstance(inst, ir.Const):
            set_use(repr(inst.value))

        elif isinstance(inst, ir.StoreVar):
            w.println(inst.name, '=', _variable(inst.value))

        elif isinstance(inst, ir.LoadVar):
            set_use(inst.name)

        elif isinstance(inst, ir.Call):

            if not inst.kwargs:
                callee = _variable(inst.callee)
                args = ', '.join(map(_variable, inst.args))
                set_use("{callee}({args})".format(callee=callee, args=args))
            else:
                assert False, 'kwargs not yet supported'

        elif isinstance(inst, ir.Op):
            binary_ops = {
                'add': '+',
                'sub': '-',
                'mul': '*',
                'lt': '<',
                'le': '<=',
                'gt': '>',
                'ge': '>=',
            }
            inplace_ops = {
                'iadd': '+='
            }
            unary_ops = {
                'bool': 'bool',
                'not': 'not',

                'iter': 'iter',
                'foriter': '__rt__.foriter',
                'foriter_valid': '__rt__.foriter_valid',
                'foriter_value': '__rt__.foriter_value',
            }
            if inst.op in binary_ops:
                assert len(inst.args) == 2
                set_use("{1} {0} {2}".format(binary_ops[inst.op],
                                             *map(_variable, inst.args)))
            elif inst.op in unary_ops:
                assert len(inst.args) == 1
                set_use("{op}({arg})".format(op=unary_ops[inst.op],
                                             arg=_variable(inst.args[0])))
            elif inst.op in inplace_ops:
                assert len(inst.args) == 2
                args = list(map(_variable, inst.args))
                w.println("{1} {0} {2}".format(inplace_ops[inst.op], *args))
                set_use(args[0])

            else:
                raise NotImplementedError(inst)
        else:
            raise NotImplementedError(inst)


def _variable(v):
    return '__var__{v.name}'.format(v=v)


def _predicate(state_or_region):
    if isinstance(state_or_region, Region):
        state = state_or_region.first
    else:
        state = state_or_region
    return '__pred__{0}'.format(state.offset)


def _descendents_map(descendents):
    out = {}
    for state in descendents:
        out[state.offset] = state
    return out


class PyCodeWriter(object):

    def __init__(self):
        self._buf = StringIO()
        self._indent = 0

    @contextmanager
    def indent(self):
        self._indent += 1
        yield
        self._indent -= 1

    def println(self, *args):
        print(' ' * (4 * self._indent), end='', file=self._buf)
        print(*args, file=self._buf)

    def show(self):
        return self._buf.getvalue()
