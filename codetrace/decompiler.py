from io import StringIO
from contextlib import contextmanager

from . import ir
from .controlflow import Region


def decompile(tracegraph, cfa):
    dec = Decompiler(tracegraph, cfa)
    return dec.decompile()


class Decompiler(object):

    def __init__(self, tracegraph, cfa):
        self._tracegraph = tracegraph
        self._cfa = cfa
        # map head of regions
        self._regionmap = {}
        regiontree = self._cfa.region_tree()
        for sr in regiontree.subregions():
            self._regionmap[sr.first] = sr

    def decompile(self):
        self._writer = PyCodeWriter()
        try:
            self._writer.println("def foo(a):")
            with self._writer.indent():
                self._decompile(self._cfa.region_tree())
            return self._writer.show()
        except:
            print(self._writer.show())
            raise

    def _decompile(self, item):
        print('decompile {0}'.format(item).center(80, '-'))
        if isinstance(item, Region):
            self._region = item  # remeber current region
            self._decompile_region(item)
        else:
            self._decompile_state(item)

    def _decompile_state(self, state):
        w = self._writer

        w.println('#', state)
        w.println(_predicate(state), '= False')

        # unpack stack
        istack = state.incoming_stack
        if istack:
            retr = ', '.join(map(_variable, istack))
            w.println('[{0}]'.format(retr), '=', _get_outgoing_stack())

        # decompile instructions (except terminator)
        insts = list(state)
        for inst in insts[:-1]:
            self._decompile_inst(state, inst)

        self._decompile_terminator(state)

    def _decompile_region(self, region):
        cfg = self._cfa.region_local_cfg(self._tracegraph, region)
        # cfg.graphviz(filename='region_local.gv')
        loops = cfg.loops()

        w = self._writer
        if not loops:
            postdoms = cfg.post_dominators()
            # determine nodes that must be at the top level
            topnodes = frozenset(postdoms[cfg.entry_point()])
            assert cfg.entry_point() in topnodes

            for node in cfg.topo_order():
                if node not in topnodes:
                    w.println(_predicate(node), '= False')

            for node in cfg.topo_order():
                if node not in topnodes:
                    w.println('#', node)
                    pred = _predicate(node)
                    w.println('if {0}:'.format(pred))
                    with w.indent():
                        self._decompile(node)
                elif node is not None:
                    w.println('#', node)
                    self._decompile(node)
        else:
            toponodes = list(filter(lambda x: x is not None, cfg.topo_order()))
            preds = list(map(_predicate, toponodes))
            for p in preds:
                w.println(p, '= False')
            w.println(_predicate(cfg.entry_point()), '= True')
            loopcond = ' or '.join(preds)
            w.println('while {cond}:'.format(cond=loopcond))
            with w.indent():
                for node in toponodes:
                    w.println('#', node)
                    pred = _predicate(node)
                    w.println('if {0}:'.format(pred))
                    with w.indent():
                        self._decompile(node)

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
                w.println(_set_outgoing_stack(term.outgoing_stack(term.then)))
            w.println('else:')
            with w.indent():
                w.println(_predicate(else_state), '= True')
                w.println(_set_outgoing_stack(
                    term.outgoing_stack(term.orelse)))

        elif isinstance(term, ir.Jump):
            target_state = self._tracegraph[state, term.target]
            w.println(_predicate(target_state), '= True')
            w.println(_set_outgoing_stack(term.outgoing_stack(term.target)))

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
            set_use(inst.value)

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
            inplace_ops = {'iadd': '+='}
            unary_ops = frozenset(['iter'])
            if inst.op in unary_ops:
                assert len(inst.args) == 1
                set_use("{op}({arg})".format(op=inst.op,
                                             arg=_variable(inst.args[0])))
            elif inst.op in inplace_ops:
                assert len(inst.args) == 2
                args = list(map(_variable, inst.args))
                w.println("{1} {0} {2}".format(inplace_ops[inst.op], *args))
                set_use(args[0])

            elif inst.op == 'foriter':
                assert len(inst.args) == 1
                w.println('try:')
                with w.indent():
                    set_use('(True, next({0}))'.format(
                        _variable(inst.args[0])))
                w.println('except StopIteration:')
                with w.indent():
                    set_use('(False, None)')

            elif inst.op == 'foriter_valid':
                assert len(inst.args) == 1
                set_use('{0}[0]'.format(_variable(inst.args[0])))

            elif inst.op == 'foriter_value':
                assert len(inst.args) == 1
                set_use('{0}[1]'.format(_variable(inst.args[0])))

            else:
                raise NotImplementedError(inst)
        else:
            raise NotImplementedError(inst)


def _variable(v):
    return '_var_{v.name}'.format(v=v)


def _predicate(state_or_region):
    if isinstance(state_or_region, Region):
        state = state_or_region.first
    else:
        state = state_or_region
    return '_pred_{0}'.format(state.offset)


def _set_outgoing_stack(values):
    return '{0} = [{1}]'.format(_get_outgoing_stack(),
                                ', '.join(map(_variable, values)))


def _get_outgoing_stack():
    return '_stack'


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
