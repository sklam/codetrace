from .state import State
from .ir import (JumpIf, Jump, Ret, LoadVar, Const, Op, StoreVar, Call,
                 Context, Block)
from .tracegraph import TraceGraph, _maybegetpc


def symbolic_evaluate(instlist):
    tracegraph = SymEval(instlist).run()
    tracegraph.verify()
    return tracegraph


class SymEval(object):

    def __init__(self, instlist):
        self._tracegraph = TraceGraph()
        self._instlist = instlist
        self._context = Context()

    def run(self):
        self._todos = {}
        self._todos[TraceGraph.first_key] = State(self._context, pc=0)

        # build pc-inst lookup table
        instmap = {}
        for pos, inst in enumerate(self._instlist):
            pc = inst['pc']
            if pc not in instmap:
                instmap[pc] = pos

        # == eval loop ==
        # Note: States are created by symbolic evaluation on the bytecode.
        #       They are checked against the existing ones for duplication.
        #       If a new state is equivalent to an existing one, it is not
        #       included.  Otherwise, the new state is added to the tracegraph.
        while self._todos:
            self._next = {}
            laststate, pc = edge = next(iter(self._todos))
            assert isinstance(laststate, State) or laststate is None
            # XXX: remove me
            print('evaluating', edge, _maybegetpc(edge[0]))
            self._state = curstate = self._todos.pop(edge)

            offset = instmap[pc]

            for inst in self._instlist[offset:]:
                # XXX: remove me
                print(inst, '|', self._state.pc, '| pc', pc)
                pc = self._state.pc = inst['pc']
                self._state.meta(kind='.loc', pc=pc)
                fname = 'op_' + inst['op']
                fn = getattr(self, fname)
                fn(inst)
                if self._state is None:
                    break

            # determine if the new state should be included
            self._determine_todos(edge, curstate)

        # XXX: temp debug
        for st in self._tracegraph.values():
            assert st._instlist
            assert st.is_terminated

        del self._next
        del self._todos
        return self._tracegraph

    def _determine_todos(self, edge, state):
        assert edge not in self._tracegraph, 'internal error: edge duplication'
        other = self._tracegraph.find_mergeable(edge, state)
        if other is None:
            # not duplicated or mergeable
            # add to _todos
            self._tracegraph[edge] = state
            self._todos.update(self._next)
        else:
            # mergeable; point edge to other
            self._tracegraph[edge] = other

    def _stop(self):
        self._state = None

    def _add_state_edge(self, state, edge):
        assert isinstance(edge[0], State)
        self._next[edge] = state

    def jump_if(self, pred, then, orelse, or_pop=False):
        then_edge = self._state, then
        else_edge = self._state, orelse
        then_args = self._state.list_stack()
        else_args = self._state.list_stack()
        if or_pop:
            else_args.pop()

        self._state.emit(JumpIf(pred, then, orelse, then_args, else_args))
        then_state = self._state.fork(pc=then)
        if or_pop:
            self._state.pop()
        else_state = self._state.fork(pc=orelse)
        self._add_state_edge(then_state, then_edge)
        self._add_state_edge(else_state, else_edge)
        self._stop()
        return then_state, else_state

    def return_value(self, value):
        self._state.emit(Ret(value))
        self._stop()

    def op_LOAD_FAST(self, inst):
        self._state.push(self._state.emit(LoadVar(inst['varname'],
                                                  scope='local')))

    def op_LOAD_GLOBAL(self, inst):
        self._state.push(self._state.emit(LoadVar(inst['name'],
                                                  scope='global')))

    def op_LOAD_CONST(self, inst):
        self._state.push(self._state.emit(Const(inst['const'])))

    def op_STORE_FAST(self, inst):
        val = self._state.pop()
        self._state.emit(StoreVar(val, inst['varname'], scope='local'))

    def op_binary(self, inst, op):
        rhs = self._state.pop()
        lhs = self._state.pop()
        res = self._state.emit(Op(op, lhs, rhs))
        self._state.push(res)

    def op_BINARY_ADD(self, inst):
        self.op_binary(inst, 'add')

    def op_BINARY_SUBTRACT(self, inst):
        self.op_binary(inst, 'sub')

    def op_inplace(self, inst, op):
        rhs = self._state.pop()
        lhs = self._state.pop()
        res = self._state.emit(Op(op, lhs, rhs))
        self._state.push(res)

    def op_INPLACE_ADD(self, inst):
        self.op_inplace(inst, 'iadd')

    def op_COMPARE_OP(self, inst):
        self.op_binary(inst, inst['compare'])

    def op_POP_JUMP_IF_FALSE(self, inst):
        val = self._state.pop()
        pred = self._state.emit(Op('bool', val))
        negated = self._state.emit(Op('not', pred))
        self.jump_if(negated, inst['to'], inst['next'])

    def op_POP_JUMP_IF_TRUE(self, inst):
        val = self._state.pop()
        pred = self._state.emit(Op('bool', val))
        self.jump_if(pred, inst['to'], inst['next'])

    def op_JUMP_IF_TRUE_OR_POP(self, inst):
        val = self._state.tos()
        pred = self._state.emit(Op('bool', val))
        self.jump_if(pred, inst['to'], inst['next'], or_pop=True)

    def op_RETURN_VALUE(self, inst):
        val = self._state.pop()
        self.return_value(val)

    def op_SETUP_LOOP(self, inst):
        self._state.push_block(Block('loop', end_loop=inst['to']))

    def op_POP_BLOCK(self, inst):
        self._state.pop_block()

    def op_CALL_FUNCTION(self, inst):
        arg = inst['arg']
        nargs = arg & 0xff
        nkwargs = (arg >> 8) & 0xff

        def pop_kwargs():
            name = self._state.pop()
            value = self._state.pop()
            return name, value

        kwargs = dict([pop_kwargs() for _ in range(nkwargs)])
        args = [self._state.pop() for _ in range(nargs)]
        callee = self._state.pop()

        retval = self._state.emit(Call(callee, args, kwargs))
        self._state.push(retval)

    def op_GET_ITER(self, inst):
        val = self._state.pop()
        out = self._state.emit(Op('iter', val))
        self._state.push(out)

    def op_FOR_ITER(self, inst):
        iterator = self._state.tos()

        iterstate = self._state.emit(Op('foriter', iterator))
        pred = self._state.emit(Op('foriter_valid', iterstate))
        iterval = self._state.emit(Op('foriter_value', iterstate))

        self._state.push(iterval)

        label_body = inst['next']
        label_end = inst['to']

        self.jump_if(pred, label_body, label_end, or_pop=True)

    def op_JUMP_ABSOLUTE(self, inst):
        to = inst['to']
        edge = self._state, to
        self._state.emit(Jump(to, self._state.list_stack()))
        self._add_state_edge(self._state.fork(to), edge)
        self._stop()

    def op_JUMP_FORWARD(self, inst):
        to = inst['to']
        edge = self._state, to
        self._state.emit(Jump(to, self._state.list_stack()))
        self._add_state_edge(self._state.fork(to), edge)
        self._stop()
