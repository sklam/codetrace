from .rewriter import Rewriter
from . import ir

from collections import defaultdict


def partial_evaluate(tracegraph):
    rewritten = ConstProp(tracegraph).rewrite()
    rewritten.verify()
    return rewritten


class ConstProp(Rewriter):
    max_cycle_limit = 1

    _constops = defaultdict(dict)

    def init(self):
        self._previous_states = {}

    def begin_state(self, incoming_state, cycle_count):
        default = dict(uses=set(), vars={})
        if cycle_count <= self.max_cycle_limit:
            d = self._previous_states.get(incoming_state, default)
        else:
            d = default
        self._const_uses = d['uses'].copy()
        self._const_vars = d['vars'].copy()
        if incoming_state is not None:
            self.meta(const_assert=self._const_vars.copy())

    def end_state(self, current_state):
        self._previous_states[current_state] = dict(uses=self._const_uses,
                                                    vars=self._const_vars)
        del self._const_uses
        del self._const_vars

    def is_constant(self, item):
        if isinstance(item, ir.Use):
            return item in self._const_uses
        else:
            return item in self._const_vars

    def visit_Const(self, inst):
        use = self.emit(inst)
        self._const_uses.add(use)

    def visit_StoreVar(self, inst):
        self.emit(inst)
        if self.is_constant(inst.value):
            self._const_vars[inst.name] = inst.value
        elif inst.name in self._const_vars:
            del self._const_vars[inst.name]

    def visit_LoadVar(self, inst):
        if self.is_constant(inst.name):
            if inst.scope == 'local':
                self.replace_with_use(self._const_vars[inst.name])
                return
        self.emit(inst)

    def visit_Op(self, inst):
        if all(map(self.is_constant, inst.args)):
            opbin = self._constops.get(inst.op)
            if opbin:
                vals = [x.value.value for x in inst.args]
                typs = tuple(map(type, vals))
                fn = opbin[typs]
                use = fn(self, vals)
                if use is not NotImplemented:
                    return
        # otherwise
        self.emit(inst)

    def visit_JumpIf(self, inst):
        if self.is_constant(inst.pred):
            pred = inst.pred.value.value
            assert type(pred) is bool
            if pred:
                self.emit(ir.Jump(inst.then, inst.then_args))
            else:
                self.emit(ir.Jump(inst.orelse, inst.else_args))
        else:
            self.emit(inst)


def constop(opname, type_spec):
    def wrap(fn):
        def run(constprop, values):
            res = fn(*values)
            if res is NotImplemented:
                return NotImplemented
            use = constprop.emit(ir.Const(res))
            constprop._const_uses.add(use)
            return use

        opbin = ConstProp._constops[opname]
        opbin[tuple(type_spec)] = run

    return wrap


@constop("gt", [int, int])
def const_gt_int(x, y):
    return x > y


@constop("eq", [int, int])
def const_eq_int(x, y):
    return x == y


@constop("iadd", [int, int])
def const_iadd_int(x, y):
    return x + y


@constop("isub", [int, int])
def const_isub_int(x, y):
    return x - y


@constop('bool', [bool])
def const_bool_bool(x):
    return x


@constop('not', [bool])
def const_not_bool(x):
    return not x
