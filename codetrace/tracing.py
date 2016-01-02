from __future__ import absolute_import, print_function
import sys
import dis
from io import StringIO
import inspect
from .bcutils import (parse_bytecode, find_jump_targets, get_code_from_object,
                      get_jump_target)
from .utils import unique_name_generator
from collections import Mapping
import weakref
import logging


logger = logging.getLogger(__name__)


def _dissemble(obj):
    """
    Call ``dis.dis(obj)`` but the output is redirected to a string
    """
    stdout = sys.stdout
    sys.stdout = buf = StringIO()
    try:
        dis.dis(obj)
    finally:
        sys.stdout = stdout
    res = buf.getvalue()
    buf.close()
    return res


def trace(obj):
    """
    Trace an object, which must have an associated code object
    Returns Traces object
    """
    # Preparation
    code = get_code_from_object(obj)
    logging.debug("code of %r:\n%s", obj, _dissemble(code))
    bcinfos = parse_bytecode(code)
    jmp_targets = find_jump_targets(bcinfos)
    entry_pt = bcinfos[0].offset

    # Build offset map
    offset_map = {}
    for bc in bcinfos:
        offset_map[bc.offset] = bc

    # Trace
    traces = run_trace(offset_map, entry_pt, jmp_targets)
    for k, v in traces.items():
        logger.info('trace %s: state %s', k, v)
        v.optimize_load_stores()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("dump state %s:\n%s", v, v.show())

    return Traces(obj, code, offset_map, traces, entry_pc=entry_pt)


class Traces(Mapping):
    """
    A dictionary-like object of traces for a single code object
    """

    def __init__(self, pyobj, pycode, offset_map, traces, entry_pc):
        self._pyobj = pyobj
        self._pycode = pycode
        self._offset_map = offset_map
        self._traces = traces
        # Additional initialization after setting basic attributes
        self._find_entry_state(entry_pc)

    def _find_entry_state(self, entry_pc):
        for label, state in self.items():
            if state.start_pc == entry_pc:
                self._entry_label = label
                self._entry_state = state
                break

    def __len__(self):
        return len(self._traces)

    def __iter__(self):
        return iter(self._traces)

    def __getitem__(self, label):
        return self._traces[label]

    def __repr__(self):
        kwargs = dict(code=self._pycode,
                      size=len(self))
        return "<Traces code={code}, # of states={size} >".format(**kwargs)

    @property
    def entry_label(self):
        return self._entry_label

    @property
    def entry_state(self):
        return self._entry_state

    @property
    def labels(self):
        return self.keys()

    @property
    def states(self):
        return self.values()

    @property
    def firstlineno(self):
        return self._pycode.co_firstlineno

    @property
    def name(self):
        return self._pycode.co_name

    @property
    def filename(self):
        return self._pycode.co_filename

    @property
    def argspec(self):
        return inspect.getargspec(self._pyobj)

    def get_dot_graph(self):
        from graphviz import Digraph

        g = Digraph(name="traces.{0}".format(self.name))
        for label, state in sorted(self.items(), key=lambda x: x[1].start_pc):
            # Add nodes
            desc = state.show(show_lifetime=False, show_meta=False)
            desc = desc.replace('\n', '\\l') + '\\l'
            g.node(str(label), '<<{0}>>\n'.format(label) + desc, shape='rect')
            # Add edges
            term = state.get_terminator()
            for name, outedge in term.get_labels().items():
                if name != 'escape':
                    g.edge(str(label), str(outedge))
            # Add exception edges
            exc_handler = state.get_except_handler()
            if exc_handler is not None:
                g.edge(str(label), str(exc_handler.escape_label),
                       style='dotted', label='except')

        return g

    def view_dot_graph(self, filename=None, view=True):
        src = self.get_dot_graph()
        if view:
            return src.render(filename, view=view)
        else:
            try:
                import IPython.display as display
            except ImportError:
                return src
            else:
                format = 'png'
                return display.Image(data=src.pipe(format))


def run_trace(offset_map, start_pc, jmp_targets):
    """
    Given `offset_map` as a mapping of bytecode offset to parsed bytecode
    objects. Start tracing at `start_pc`.  Using `jmp_targets` as a sequence
    of starting offset of new traces.
    """
    symexec = SymbolicExecutor()
    state = TraceState(start_pc=start_pc)
    heads = [state]
    traces = {}
    labels = {state: Label()}
    label_repl = {}
    # For each trace heads
    while heads:
        state = heads.pop()
        key = state.fingerprint
        # Skip if the trace has been executed
        if key in traces:
            label_repl[labels[state]] = labels[traces[key]]
            continue

        traces[key] = state
        pc = state.start_pc
        logger.debug("tracing PC=%s", pc)
        while True:
            cur_bc = offset_map[pc]
            signal = symexec.execute(state, cur_bc)
            pc_or_none = _handle_signals(state, jmp_targets, heads, labels,
                                         cur_bc, pc, signal)
            if pc_or_none is None:
                break
            else:
                pc = pc_or_none

    # Replace labels
    for st in traces.values():
        st.replace_labels(label_repl)

    return dict((labels[st], st) for st in traces.values())


def _handle_signals(state, jmp_targets, heads, labels, cur_bc, pc, signal):
    # Continue to the next bytecode
    if isinstance(signal, Continue):
        # Next position is actually a trace head
        if cur_bc.next in jmp_targets:
            target = cur_bc.next

            label = Label()
            state.code_append("jump", target=label)

            newstate = state.fork(label, target)
            heads.append(newstate)
            labels[newstate] = label
            return
        # Goto next bytecode
        else:
            pc = cur_bc.next
            return pc
    # Current trace has ended
    elif isinstance(signal, StopTrace):
        return

    # Current trace reaches a branch
    elif isinstance(signal, Branch):
        for label, (target, extra) in signal.iter_target_items():
            if extra is None:
                newstate = state.fork(label, target)
            else:
                newstate = state.fork(label, target, extra)
            heads.append(newstate)
            labels[newstate] = label
        return

    elif isinstance(signal, Except):
        try_state = state.fork(signal.try_block[0], signal.try_block[1])
        exceptinfos = [Traceback(),
                       ExceptionValue(),
                       ExceptionType(), ]
        exc_state = state.fork(signal.except_block[0],
                               signal.except_block[1],
                               extra_stack=exceptinfos,
                               pop_block=True,
                               exceptional=True)
        heads.append(try_state)
        heads.append(exc_state)
        labels[try_state] = signal.try_block[0]
        labels[exc_state] = signal.except_block[0]
        return

    elif isinstance(signal, Finally):
        # XXX: can we not special case the With signal. it is the same?
        if isinstance(signal, With):
            try_state = state.fork(signal.try_block[0],
                                   signal.try_block[1])
            exceptinfos = [Traceback(),
                           ExceptionValue(),
                           ExceptionType(), ]
            fin_state = state.fork(signal.finally_block[0],
                                   signal.finally_block[1],
                                   extra_stack=exceptinfos,
                                   pop_block=True)
            heads.append(try_state)
            heads.append(fin_state)
            labels[try_state] = signal.try_block[0]
            labels[fin_state] = signal.finally_block[0]
            return
        else:
            try_state = state.fork(signal.try_block[0],
                                   signal.try_block[1])
            exceptinfos = [Traceback(),
                           ExceptionValue(),
                           ExceptionType(), ]
            fin_state = state.fork(signal.finally_block[0],
                                   signal.finally_block[1],
                                   extra_stack=exceptinfos,
                                   pop_block=True)
            heads.append(try_state)
            heads.append(fin_state)
            labels[try_state] = signal.try_block[0]
            labels[fin_state] = signal.finally_block[0]
            return

    elif isinstance(signal, Raise):
        return
    # ???
    else:
        raise TypeError("invalid signal type: {0}".format(type(signal)))


class TraceState(object):
    """
    Records the stack, block stack (frame block) and instructions.
    """

    def __init__(self, start_pc, incoming_stack=(), incoming_block_stack=()):
        self.start_pc = start_pc
        self.incoming_stack = tuple(incoming_stack)
        self._stack = list(incoming_stack)
        self._block_stack = list(incoming_block_stack)
        self.code = []
        self._fingerprint = (start_pc, len(incoming_stack),
                             len(incoming_block_stack))
        self._outgoing_stacks = {}

        # Find exception handler block (e.g. ExceptBlock or FinallyBlock)
        self._except_handler = None
        for blk in reversed(self._block_stack):
            if isinstance(blk, OutOfBandBlock) and blk.in_range(start_pc):
                self._except_handler = blk
                break

        # Find finally block (e.g. for return)
        self._finally_handler = None
        for blk in reversed(self._block_stack):
            if isinstance(blk, FinallyBlock) and blk.in_range(start_pc):
                self._finally_handler = blk
                break

    @property
    def fingerprint(self):
        return self._fingerprint

    def fork(self, label, pc, extra_stack=(), pop_block=False,
             exceptional=False):
        """
        Use when the state branches.
        The `pc` argument indicates the new starting program counter.
        The `extra_stack` argument is used to pass extra values that will go
        to the end of the stack of the new trace state.
        if the `exceptional` argument is set, the target branch is not
        considered a outgoing branch.

        Returns a new trace state.
        The new trace state inherits the stack and block stack.
        The code list is cleared.
        """
        assert isinstance(label, Label)
        incoming_stack = list(self._stack)
        incoming_stack.extend(extra_stack)
        incoming_block_stack = self._block_stack
        if pop_block:
            incoming_block_stack = incoming_block_stack[:-1]
        assert label not in self._outgoing_stacks
        if not exceptional:
            self._outgoing_stacks[label] = tuple(incoming_stack)
        return TraceState(pc, incoming_stack, incoming_block_stack)

    @property
    def outgoing_values(self):
        return set(v for stacks in self._outgoing_stacks.values()
                   for v in stacks)

    @property
    def outgoing_stacks(self):
        return iter(self._outgoing_stacks.items())

    def outgoing_stack(self, label):
        return self._outgoing_stacks[label]

    @property
    def outoing_labels(self):
        return set(self._outgoing_stacks.keys())

    def stack_pop(self):
        return self._stack.pop()

    def stack_push(self, val):
        assert isinstance(val, Value)
        self._stack.append(val)

    def stack_peek(self):
        return self._stack[-1]

    @property
    def is_stack_empty(self):
        return not self._stack

    def block_push(self, block):
        self._block_stack.append(block)

    def block_pop(self):
        return self._block_stack.pop()

    def block_peek(self):
        return self._block_stack[-1]

    def code_append(self, opcode, **operands):
        """
        Append a new instruction to the code attribute.
        Operands are specified as keyword arguments.
        The "out" keyword is reserved for the output of the instruction.
        The "out" keyword is not required.
        """
        inst = Inst(opcode, **operands)
        self.code.append(inst)
        return inst

    def get_terminator(self):
        return self.code[-1]

    def get_except_handler(self):
        if not self._except_handler:
            return None
        else:
            return self._except_handler

    def get_finally_handler(self):
        if not self._finally_handler:
            return None
        else:
            return self._finally_handler

    def replace_labels(self, mapping):
        for inst in self.code:
            inst.replace_labels(mapping)
        new_out_stacks = {}
        for label, stack in self.outgoing_stacks:
            new_out_stacks[mapping.get(label, label)] = stack
        self._outgoing_stacks = new_out_stacks

    def show(self, show_lifetime=True, show_meta=True):
        """
        Returns a string representation of this trace state.
        """
        buf = ["PC = {0}".format(self.start_pc)]
        buf.append("incoming stack = {0}".format(self.incoming_stack))
        for label, stack in self.outgoing_stacks:
            buf.append("outgoing {0} = {1}".format(label, stack))
        buf.append("block stack = {0}".format(self._block_stack))
        if self._except_handler:
            fmt = "except handler = {0}"
            buf.append(fmt.format(self.get_except_handler().escape_label))

        for inst, livevars in zip(self.code, self.compute_lifetime()):
            if not show_meta:
                if inst.opcode.startswith('.'):
                    continue

            parts = dict(inst=str(inst),
                         alive=', '.join(map(str, livevars)))

            fmt = "  {inst:60s}"
            if show_lifetime and livevars:
                fmt += "   # {alive}"

            buf.append(fmt.format(**parts))
        return '\n'.join(buf)

    def compute_lifetime(self):
        """
        Returns a list of active values (as a set) at the point of each
        instruction. The entries in this corresponds to the instruction at
        the same position in `self.code`.
        """
        out = []
        active = set(filter(lambda x: isinstance(x, Value),
                            self.outgoing_values))
        for inst in reversed(self.code):
            active |= set(inst.get_operand_values().values())
            out.append(frozenset(active))
            active.discard(inst.out)
        return list(reversed(out))

    def optimize_load_stores(self):
        """
        Optimize the code to eliminate unncessary load and stores.

        Note: We cannot eliminate any stores to preserve all side-effects in
              case of the trace has a side-escape due to branching or
              exceptions.
        """
        last_store = {}
        last_load = {}

        dead_load = set()

        replace_values = {}
        for inst in self.code:
            if inst.opcode == 'load':
                name = inst.operands['src']
                out = inst.out

                # A load can be eliminated if the content is previously stored
                # within this trace.
                if name in last_store:
                    dead_load.add(inst)

                    stored_val = last_store[name].operands['src']
                    replace_values[out] = stored_val

                # A load can be eliminated if the content is already loaded
                # and there were no store that invalidated the previous load
                elif name in last_load:
                    dead_load.add(inst)
                    replace_values[out] = last_load[name].out

                # Remember this load
                else:
                    last_load[name] = inst

            elif inst.opcode == 'store':
                name = inst.operands['dst']
                # Overwrite previous store
                last_store[name] = inst
                # Invalidate previous load
                last_load.pop(name, None)

        # Rewrite the code
        # - eliminate dead load
        # - replace all usage of the result of dead loads

        def rewrite_gen():
            for inst in self.code:
                if inst.opcode == 'load':
                    if inst not in dead_load:
                        yield inst

                else:
                    inst.replace_operands(replace_values)
                    yield inst

        self.code = list(rewrite_gen())


class Inst(object):
    def __init__(self, opcode, **operands):
        self.opcode = opcode
        self.out = operands.pop('out', None)
        self.operands = operands
        if self.out is not None:
            self.out.set_source(self)

    def __repr__(self):
        kwargs = dict(opcode=self.opcode,
                      operands=', '.join("{0}={1}".format(k, self.operands[k])
                                         for k in sorted(self.operands)))
        if self.out is None:
            return "{opcode} {operands}".format(**kwargs)
        else:
            kwargs['out'] = self.out
            return "{out} <- {opcode} {operands}".format(**kwargs)

    def replace_operands(self, repl):
        for k, v in self.operands.items():
            self.operands[k] = repl.get(v, v)

    def get_operand_values(self):
        """
        Returns all Value objects in the operands (excluding out)
        """
        return dict(filter(lambda x: isinstance(x[1], Value),
                           self.operands.items()))

    def replace_labels(self, repl):
        for k, v in self.operands.items():
            if isinstance(v, Label) and v in repl:
                self.operands[k] = repl[v]

    def get_labels(self):
        return dict(filter(lambda x: isinstance(x[1], Label),
                           self.operands.items()))


class Block(object):
    def __init__(self):
        self._source = None
        self._name = _unique_namer()

    def set_source(self, inst):
        assert 'end' in inst.operands, 'inst does not define "end"'
        self._source = weakref.ref(inst)

    @property
    def source(self):
        return self._source()

    def __repr__(self):
        return "@{0}.{1}".format(type(self).__name__, self._name)

    @property
    def end_offset(self):
        """
        Ending bytecode offset
        """
        return self.source.operands['end']

    def in_range(self, pc):
        return self.end_offset >= pc


class LoopBlock(Block):
    pass


class OutOfBandBlock(Block):
    @property
    def escape_label(self):
        return self.source.operands['escape']


class ExceptBlock(OutOfBandBlock):
    pass


class FinallyBlock(OutOfBandBlock):
    pass


_unique_name_iter = iter(unique_name_generator())
_unique_namer = lambda: next(_unique_name_iter)


class Value(object):
    pass


class TempValue(Value):
    prefix = 'tmp'

    def __init__(self):
        self._source = None
        self._name = _unique_namer()

    @property
    def name(self):
        return self._name

    def set_source(self, inst):
        self._source = weakref.ref(inst)

    @property
    def source(self):
        if self._source is None:
            raise ValueError("source is not defined")
        src = self._source()
        if src is None:
            raise ValueError("source instruction is released")
        return src

    def __repr__(self):
        return "%{0}.{1}".format(self.prefix, self._name)


class Returned(TempValue):
    prefix='ret'


class Slienced(TempValue):
    prefix = "slienced"


class InjectedValue(Value):
    prefix = None  # to be overriden in subclass

    def __init__(self):
        self._name = _unique_namer()

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return "%{0}.{1}".format(self.prefix, self._name)



class ExceptionType(InjectedValue):
    prefix = 'excepttype'


class ExceptionValue(InjectedValue):
    prefix = 'exceptvalue'


class Traceback(InjectedValue):
    prefix = 'traceback'


class Label(object):
    """
    A jump label.
    Note: instances of Label must be immutable.
    """

    def __init__(self, _name=None):
        self._name = _unique_namer() if _name is None else _name

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return "label.{0}".format(self.name)

    def __eq__(self, other):
        return isinstance(other, Label) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


class TraceSignal(object):
    """
    Signals returned by SymbolicExecutor.execute to indicate different
    "next action'.
    """
    pass


class StopTrace(TraceSignal):
    """
    Stop the trace because the code has returned.
    """
    pass


class Continue(TraceSignal):
    """
    Continue the trace to the next bytecode instruction.
    """
    pass


class Raise(TraceSignal):
    """
    Trace stop due to exception thrown
    """
    pass


class Branch(TraceSignal):
    """
    The trace branches to different locations.
    """

    def __init__(self):
        self._labels = {}

    def add_target(self, label, target, extra=None):
        assert label not in self._labels
        assert isinstance(extra, (type(None), tuple, list))
        self._labels[label] = (target, extra)

    def iter_target_items(self):
        return self._labels.items()


class Except(TraceSignal):
    """
    Indicates an exception signal
    """

    def __init__(self, try_block, except_block):
        self.try_block = try_block
        self.except_block = except_block


class Finally(TraceSignal):
    """
    Indicates a finally signal
    """

    def __init__(self, try_block, finally_block):
        self.try_block = try_block
        self.finally_block = finally_block


class With(Finally):
    """
    Indicate a with signal
    """


class SymbolicExecutor(object):
    """
    Symbolically execute a bytecode and records the action into the
    trace state.

    Handling of each bytecode is defined under op_XXX methods.
    """

    def execute(self, state, bc):
        """
        Returns different values to indicate the next action:
          - None to proceed to the next PC (bc.next)
          - Sequence of jump targets
          - StopIteration to end the trace
        """
        state.code_append(".loc", offset=bc.offset)
        fn = getattr(self, "op_{0}".format(bc.opname))
        logger.debug("translating %s", bc)
        signal = fn(state, bc)
        assert isinstance(signal, TraceSignal)
        return signal

    def op_POP_TOP(self, state, bc):
        state.stack_pop()
        return Continue()

    def op_DUP_TOP(self, state, bc):
        val = state.stack_peek()
        state.stack_push(val)
        return Continue()

    def op_LOAD_ATTR(self, state, bc):
        val = state.stack_pop()
        res = TempValue()
        state.code_append("get_attr", out=res, src=val, attr=bc.args[0])
        state.stack_push(res)
        return Continue()

    def op_LOAD_CONST(self, state, bc):
        res = TempValue()
        state.code_append("const", out=res, value=bc.args[0])
        state.stack_push(res)
        return Continue()

    def op_LOAD_GLOBAL(self, state, bc):
        res = TempValue()
        state.code_append("global", out=res, name=bc.args[0])
        state.stack_push(res)
        return Continue()

    def op_LOAD_FAST(self, state, bc):
        varname = bc.args[0]
        res = TempValue()
        state.code_append("load", out=res, src=varname)
        state.stack_push(res)
        return Continue()

    def op_STORE_FAST(self, state, bc):
        varname = bc.args[0]
        val = state.stack_pop()
        state.code_append("store", src=val, dst=varname)
        return Continue()

    def op_binary(self, state, bc, opcode):
        rhs = state.stack_pop()
        lhs = state.stack_pop()
        res = TempValue()
        state.code_append(opcode, out=res, lhs=lhs, rhs=rhs)
        state.stack_push(res)
        return Continue()

    def op_BINARY_ADD(self, state, bc):
        return self.op_binary(state, bc, opcode="add")

    def op_BINARY_MULTIPLY(self, state, bc):
        return self.op_binary(state, bc, opcode="mul")

    def op_COMPARE_OP(self, state, bc):
        cmpop = {'<': 'lt',
                 '>': 'gt',
                 'exception match': 'exceptmatch'}[bc.args[0]]
        return self.op_binary(state, bc, opcode="cmp_{0}".format(cmpop))

    def op_jump_if(self, state, bc, jump_if, pop_after=False):
        val = state.stack_pop()
        target = get_jump_target(bc)

        if_true, if_false = (target, bc.next) if jump_if else (bc.next, target)

        labels = {True: Label(), False: Label()}
        state.code_append("jump_if", cond=val, target_true=labels[True],
                          target_false=labels[False])

        signal = Branch()
        if pop_after:
            signal.add_target(labels[jump_if], target, extra=[val])
        else:
            signal.add_target(labels[jump_if], target)
        signal.add_target(labels[not jump_if], bc.next)
        return signal

    def op_POP_JUMP_IF_FALSE(self, state, bc):
        return self.op_jump_if(state, bc, jump_if=False)

    def op_JUMP_IF_TRUE_OR_POP(self, state, bc):
        return self.op_jump_if(state, bc, jump_if=True, pop_after=True)

    def op_JUMP_FORWARD(self, state, bc):
        target = get_jump_target(bc)

        label = Label()
        state.code_append("jump", target=label)

        signal = Branch()
        signal.add_target(label, target)
        return signal

    def op_JUMP_ABSOLUTE(self, state, bc):
        target = get_jump_target(bc)
        label = Label()
        state.code_append("jump", target=label)

        signal = Branch()
        signal.add_target(label, target)
        return signal

    def op_return(self, state, val):
        fini = state.get_finally_handler()
        if fini is not None:
            state.stack_push(val)
            ret = Returned()
            state.code_append("set_returned_value", value=val, out=ret)
            state.stack_push(ret)
            label = Label()
            state.code_append("jump", target=label)
            # clear all the blocks until the finally block
            while not isinstance(state.block_peek(), FinallyBlock):
                state.block_pop()
            # pop the finally block as well
            state.block_pop()

            signal = Branch()
            signal.add_target(label, fini.end_offset)
            return signal
        else:
            state.code_append("ret", value=val)
            return StopTrace()

    def op_RETURN_VALUE(self, state, bc):
        val = state.stack_pop()
        return self.op_return(state, val)

    def op_FOR_ITER(self, state, bc):
        iterator = state.stack_peek()
        ind = TempValue()
        target = get_jump_target(bc)

        target_label = Label()
        next_label = Label()
        state.code_append("for_iter", out=ind, iter=iterator,
                          loop_exit=target_label, loop_entry=next_label)

        signal = Branch()
        signal.add_target(next_label, bc.next, extra=[ind])
        signal.add_target(target_label, target)
        return signal

    def op_POP_BLOCK(self, state, bc):
        block = state.block_pop()
        state.code_append("pop_block", block=block)
        if isinstance(block, LoopBlock):
            state.stack_pop()
        label = Label()
        state.code_append("jump", target=label)

        signal = Branch()
        signal.add_target(label, bc.next)
        return signal

    def op_POP_EXCEPT(self, state, bc):
        return Continue()

    def op_SETUP_LOOP(self, state, bc):
        block = LoopBlock()
        state.code_append("push_block",
                          kind='loop',
                          end=get_jump_target(bc),
                          out=block)
        state.block_push(block)
        return Continue()

    def op_CALL_FUNCTION(self, state, bc):
        npos, nkey = bc.args
        keynames = []
        keyvalues = []
        for _ in range(nkey):
            k = state.stack_pop()
            v = state.stack_pop()
            keynames.append(k)
            keyvalues.append(v)
        args = [state.stack_pop() for _ in range(npos)]
        func = state.stack_pop()
        res = TempValue()

        args_tup = TempValue()
        state.code_append("prepare_args", out=args_tup)
        for arg in reversed(args):
            new_args_tup = TempValue()
            state.code_append("append_args", args=args_tup, value=arg,
                              out=new_args_tup)
            args_tup = new_args_tup

        if nkey:
            kwargs_dict = TempValue()
            state.code_append("build_dict", out=kwargs_dict)
            for i, (k, v) in enumerate(zip(keynames, keyvalues)):
                state.code_append("dict_setitem", dict=kwargs_dict, key=k,
                                  value=v)
        else:
            kwargs_dict = None

        if kwargs_dict:
            state.code_append("call", out=res, func=func,
                              args=args_tup, kwargs=kwargs_dict)
        else:
            state.code_append("call", out=res, func=func, args=args_tup)

        state.stack_push(res)
        return Continue()

    def op_GET_ITER(self, state, bc):
        res = TempValue()
        state.code_append("get_iter", out=res, iter=state.stack_pop())
        state.stack_push(res)
        return Continue()

    def op_SETUP_EXCEPT(self, state, bc):
        block = ExceptBlock()
        target = get_jump_target(bc)
        next_label = Label()
        target_label = Label()
        state.code_append("push_block", kind="except",
                          target=next_label,
                          escape=target_label,
                          end=target,
                          out=block)
        state.block_push(block)
        state.code_append("jump", target=next_label)
        return Except(try_block=(next_label, bc.next),
                      except_block=(target_label, target))

    def op_END_FINALLY(self, state, bc):
        def is_const_none(val):
            inst = val.source
            return inst.opcode == 'const' and inst.operands['value'] is None

        if state.is_stack_empty:
            raise RuntimeError("stack is empty at END_FINALLY", bc)
        elif isinstance(state.stack_peek(), ExceptionType):
            et = state.stack_pop()  # exception type
            ev = state.stack_pop()  # exception value
            tb = state.stack_pop()  # traceback
            state.code_append("reraise", type=et, value=ev, traceback=tb)
            return Raise()
        elif isinstance(state.stack_peek(), Slienced):
            state.stack_pop()
            return Continue()
        elif isinstance(state.stack_peek(), Returned):
            state.stack_pop()
            state.code_append("ret", value=state.stack_pop())
            return StopTrace()
        elif is_const_none(state.stack_peek()):
            state.stack_pop()
            return Continue()
        else:
            template = "unhandled case in END_FINALLY: {0}"
            raise TypeError(template.format(state.stack_peek().source))

    def op_SETUP_FINALLY(self, state, bc):
        block = FinallyBlock()
        target = get_jump_target(bc)
        next_label = Label()
        target_label = Label()
        state.code_append("push_block", kind="finally",
                          target=next_label,
                          escape=target_label,
                          end=target,
                          out=block)
        state.block_push(block)
        state.code_append("jump", target=next_label)
        return Finally(try_block=(next_label, bc.next),
                       finally_block=(target_label, target))

    def op_SETUP_WITH(self, state, bc):
        block = FinallyBlock()
        tos = state.stack_pop()

        exitfn = TempValue()
        state.code_append("with_load_exit", src=tos, out=exitfn)
        state.stack_push(exitfn)

        res = TempValue()
        state.code_append("with_call_enter", src=tos, out=res)
        state.stack_push(res)

        target = get_jump_target(bc)

        next_label = Label()
        target_label = Label()
        state.code_append("push_block", kind="with",
                          target=next_label,
                          escape=target_label,
                          end=target,
                          out=block)
        state.block_push(block)
        return With(try_block=(next_label, bc.next),
                    finally_block=(target_label, target))

    def op_WITH_CLEANUP(self, state, bc):
        if isinstance(state.stack_peek(), ExceptionType):
            et = state.stack_pop()
            ev = state.stack_pop()
            tb = state.stack_pop()
            state.stack_pop()  # result of __enter__
            exit = state.stack_pop()

            res = TempValue()
            state.code_append("with_exit_raised", exit=exit, type=et, value=ev,
                              traceback=tb, out=res)

            slienced = Slienced()
            state.code_append("slienced", out=slienced)

            true_label = Label()
            false_label = Label()
            state.code_append("jump_if", cond=res, target_true=true_label,
                              target_false=false_label)

            br = Branch()
            br.add_target(true_label, bc.next, extra=[slienced])
            br.add_target(false_label, bc.next, extra=[tb, ev, et])
            return br
        else:
            tos = state.stack_peek()
            tos_src = tos.source
            if tos_src.opcode == 'const' and tos_src.operands['value'] is None:
                state.stack_pop()
                exit = state.stack_pop()
                state.code_append("with_exit_normal", exit=exit)
                state.stack_push(Slienced())
                return Continue()

        raise NotImplementedError("unknown case in WITH_CLEANUP")
