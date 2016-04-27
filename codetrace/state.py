from .ir import Block, Use, Value, Meta, Terminator, Placeholder, Inst, Jump


class MalformStateError(ValueError):
    pass


class State(object):

    def __init__(self, context, pc):
        self._context = context
        self._init_pc = pc
        self._init_stack = ()
        self._stack = []
        self._blocks = []
        self._instlist = []
        self._pc = pc

    def __len__(self):
        return self._instlist

    def __iter__(self):
        return iter(self._instlist)

    @property
    def offset(self):
        return self._init_pc

    @property
    def pc(self):
        return self._pc

    @pc.setter
    def pc(self, pc):
        self._pc = pc

    @property
    def terminator(self):
        return self._instlist[-1]

    @property
    def is_terminated(self):
        return self._instlist and isinstance(self._instlist[-1], Terminator)

    @property
    def incoming_stack(self):
        return self._init_stack

    def outgoing_labels(self):
        return [(self, label) for label in self.terminator.outgoing_labels()]

    def outgoing_stack(self, label):
        return self.terminator.outgoing_stack(label)

    def list_stack(self):
        return list(self._stack)

    def branch_stack_agree(self, otherstate):
        """
        Check if the stack agrees at the branch
        """
        nout = len(self.outgoing_stack(otherstate.offset))
        nin = len(otherstate.incoming_stack)
        return nout >= nin

    def push(self, value):
        assert isinstance(value, Use)
        self._stack.append(value)

    def pop(self):
        return self._stack.pop()

    def tos(self):
        tos = self.pop()
        self.push(tos)
        return tos

    def push_block(self, block):
        assert isinstance(block, Block)
        self._blocks.append(block)

    def pop_block(self):
        return self._blocks.pop()

    def emit(self, inst):
        assert isinstance(inst, Inst)
        assert not self.is_terminated
        if isinstance(inst, Value):
            inst = self._context.make_use(inst)
        self._instlist.append(inst)
        return inst

    def meta(self, **kwargs):
        self.emit(Meta(**kwargs))

    def fork(self, pc):
        newstate = object.__new__(State)
        newstate._context = self._context
        newstate._init_pc = pc

        newstate._init_stack = []
        newstate._stack = []
        newstate._blocks = list(self._blocks)
        newstate._instlist = []
        newstate._pc = self._pc

        # insert placeholder for leftover values on the stack
        newstate._make_stack_placeholder(len(self._stack))
        for it in newstate._init_stack:
            newstate.push(it)
        return newstate

    def clone_for_rewrite(self):
        assert self.is_terminated
        newstate = object.__new__(State)
        newstate._context = self._context
        newstate._init_pc = self._init_pc

        newstate._init_stack = []
        newstate._stack = []
        newstate._blocks = list(self._blocks)
        newstate._instlist = []
        newstate._pc = self._init_pc

        newstate._make_stack_placeholder(len(self._init_stack))
        return newstate

    def _make_stack_placeholder(self, count):
        for i in range(count - 1, -1, -1):
            ph = Placeholder(name='tos{0}'.format(i))
            use = self._context.make_use(ph)
            self._init_stack.append(use)

    def show(self):
        buf = ['--- State pc={0} ---'.format(self._init_pc)]
        buf.append('init stack: {0}'.format(self._init_stack))
        buf.append('-' * len(buf[0]))
        for inst in self._instlist:
            buf.append(inst.show())
        return '\n'.join(buf)

    def verify(self):
        # has terminator
        if not isinstance(self.terminator, Terminator):
            msg = 'missing terminator in state {0}'.format(self._init_pc)
            raise MalformStateError(msg)

    def can_merge(self, other):
        # XXX CHECK stack size may not match!?!?
        # if len(self._init_stack) != len(other._init_stack):
        #     return False
        if len(self._instlist) != len(other._instlist):
            return False
        for ai, bi in zip(self._instlist, other._instlist):
            if not ai.equivalent(bi):
                return False
        return True

    def combine(self, latter):
        assert self._context is latter._context
        assert self.is_terminated
        assert latter.is_terminated
        assert isinstance(self.terminator, Jump)
        # build mapping to fix latter's placeholders
        repl = _PassThruDict()

        for ph, val in zip(latter._init_stack, self.terminator.args):
            repl[ph] = val

        def repluses(inst):
            if isinstance(inst, Use):
                inst.replace(repluses(inst.value))
                return inst
            elif hasattr(inst, 'replace_uses'):
                return inst.replace_uses(repl)
            else:
                return inst

        self._stack = list(latter._stack)
        self._blocks = list(latter._blocks)
        self._instlist.pop()  # remove terminator
        self._instlist.extend(map(repluses, latter._instlist))
        self._pc = latter._pc


class _PassThruDict(dict):

    def __new__(cls):
        return dict.__new__(cls)

    def __getitem__(self, k):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            return k
