import logging
import builtins
from collections import defaultdict

from .. import ir, rewriter


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def constant_propagation(tracegraph):
    rewritten = ConstProp(tracegraph).rewrite()
    rewritten.verify()
    return rewritten


class ConstData(object):
    def init(self):
        return dict(uses=set(), vars={})

    def copy(self):
        return dict(uses=self.uses.copy(), vars=self.vars.copy())

    def meta(self):
        return dict(const_assert=self.vars.copy())


class ConstProp(rewriter.Rewriter):
    max_cycle_limit = 1

    immutable_globals = frozenset(['isinstance', 'int', 'float'])

    _constops = defaultdict(dict)

    def get_state_data(self):
        dct = super(ConstProp, self).get_state_data()
        assert 'const' not in dct
        dct.update({'const': ConstData})
        return dct

    def is_constant(self, item):
        if isinstance(item, ir.Use):
            return item in self.data.const.uses
        else:
            return item in self.data.const.vars

    def add_constant_uses(self, inst):
        use = self.emit(inst)
        self.data.const.uses.add(use)
        logger.debug('add const_uses %s -> %s', use, use.value)
        return use

    def visit_Const(self, inst):
        self.add_constant_uses(inst)

    def visit_StoreVar(self, inst):
        self.emit(inst)
        if self.is_constant(inst.value):
            self.data.const.vars[inst.name] = inst.value
        elif inst.name in self.data.const.vars:
            del self.data.const.vars[inst.name]

    def visit_LoadVar(self, inst):
        if inst.scope == 'local':
            if self.is_constant(inst.name):
                self.replace_with_use(self.data.const.vars[inst.name])
                return

        elif inst.scope == 'global':
            if inst.name in self.immutable_globals:
                self.add_constant_uses(inst)
                self.data.const.vars[inst.name] = getattr(builtins, inst.name)
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
            constprop.data.const.uses.add(use)
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
