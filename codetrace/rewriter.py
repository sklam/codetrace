from .tracegraph import TraceGraph
from .state import State
from . import ir

from collections import defaultdict


class Rewriter(object):

    def __init__(self, old):
        self.__old = old
        self.__new = TraceGraph()
        self.__usemap = {}
        self.init()

    def init():
        pass

    def begin_state(self, incoming_state, cyclic):
        pass

    def end_state(self, current_state):
        pass

    def meta(self, **kwargs):
        self.__curstate.meta(**kwargs)

    def emit(self, inst):
        use = self.__curstate.emit(inst)
        if isinstance(use, ir.Use):
            self.__usemap[self.__curuse] = use
            self.__curuse = None
        return use

    def replace_with_use(self, use):
        assert isinstance(use, ir.Use)
        assert self.__curuse is not None
        self.__usemap[self.__curuse] = use
        self.__curuse = None

    def rewrite(self):
        pending = set([self.__old.first_key])
        new2old = {None: None}
        cycle = defaultdict(int)

        while pending:
            st, pc = edge = pending.pop()

            oldstate = self.__old[new2old[st], pc]
            newstate = oldstate.clone_for_rewrite()
            new2old[newstate] = oldstate

            self.__curstate = newstate

            # record uses from incoming stack
            assert len(oldstate.incoming_stack) == len(newstate.incoming_stack)
            for old, new in zip(oldstate.incoming_stack,
                                newstate.incoming_stack):
                self.__usemap[old] = new

            # rewrite to new state
            self.begin_state(st, cyclic=(cycle[oldstate] > 5))
            for inst in oldstate:
                self._dispatch(inst)
            self.end_state(newstate)
            assert newstate.is_terminated
            outedges = newstate.outgoing_labels()

            # insert new state
            other = self.__new.find_mergeable(edge, newstate)
            if other is None:
                cycle[oldstate] += 1
                self.__new[edge] = newstate
                pending |= set(outedges)
            else:
                self.__new[edge] = other

        return self.__new

    def _dispatch(self, inst):
        if isinstance(inst, ir.Use):
            self.__curuse = inst
            inst = inst.value
        else:
            self.__curuse = None
            if isinstance(inst, ir.Meta) and inst.values.get('kind') == '.loc':
                self.__curstate.pc = inst.values['pc']
        self._visit(self.replace_uses(inst))

    def _visit(self, inst):
        fname = 'visit_' + inst.__class__.__name__
        fn = getattr(self, fname, self.visit_generic)
        return fn(inst)

    def replace_uses(self, inst):
        if hasattr(inst, 'replace_uses'):
            return inst.replace_uses(self.__usemap)
        else:
            return inst

    def visit_generic(self, inst):
        self.emit(inst)
