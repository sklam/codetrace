import logging
from collections import defaultdict

from .. import ir
from . import constprop


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def type_inference(tracegraph, typeinfos):
    rewritten = TypeInferer(tracegraph, typeinfos=typeinfos).rewrite()
    rewritten.verify()
    return rewritten


class TypeInfoData(object):

    def init(self):
        return dict(typeinfos={})

    def copy(self):
        return dict(typeinfos=self.typeinfos.copy())

    def meta(self):
        return dict(type_assert=self.typeinfos.copy())


class TypeInferer(constprop.ConstProp):
    max_cycle_limit = 1

    def get_state_data(self):
        dct = super(TypeInferer, self).get_state_data()
        assert 'typeinfer' not in dct
        dct.update({'typeinfer': TypeInfoData})
        return dct

    def begin_state(self, incoming_state, cycle_count):
        super(TypeInferer, self).begin_state(incoming_state, cycle_count)
        # is first state
        if incoming_state is None:
            self.data.typeinfer.typeinfos.update(self.kwargs['typeinfos'])

    def visit_Call(self, inst):
        if self.is_constant(inst.callee):
            loadcallee = inst.callee.value
            if (isinstance(loadcallee, ir.LoadVar) and
                    loadcallee.name == 'isinstance' and
                    loadcallee.scope == 'global'):
                if len(inst.args) == 2 and not inst.kwargs:
                    val, typ = inst.args
                    loadinst = val.value
                    if isinstance(loadinst, ir.LoadVar):
                        known_type = self.data.typeinfer.typeinfos[loadinst.name]
                        # XXX: constant does not imply type.
                        if self.is_constant(typ):
                            if self.data.const.vars[typ.value.name] == known_type:
                                self.add_constant_uses(ir.Const(True))
                                return

        return super(TypeInferer, self).visit_Call(inst)
