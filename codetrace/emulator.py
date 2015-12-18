"""
Generate python source code that can execute a trace.
The main purpose is verification of our static analysis.
"""
import sys
from contextlib import contextmanager
from six import exec_


class Emulator(object):
    def __init__(self, traces):
        fname, src = generate_source(traces, SourceBuffer())
        self.source_code = src
        scope = {mangle('VM_CTOR'): VirtualMachine}
        exec_(self.source_code, scope)
        self.function = scope[fname]

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def generate_source(traces, srcbuf):
    funcname = mangle("emulate_{0}".format(traces.name))
    deftext = format_def(funcname,
                         format_argspec(traces.argspec))
    with srcbuf.indent_context(deftext):
        generate_state_machine(traces, srcbuf)
    return funcname, srcbuf.show()


class VirtualMachine(object):
    def __init__(self, locals_dict, globals_dict):
        self.locals = locals_dict.copy()
        self.globals = globals_dict
        self.stack = []
        self._pc = 0
        self._handler_stack = []

    @property
    def pc(self):
        return self._pc

    @pc.setter
    def pc(self, val):
        self._pc = val

    def store(self, val, name):
        self.locals[name] = val

    def load(self, name):
        return self.locals[name]

    def push(self, val):
        self.stack.append(val)

    def pop(self):
        return self.stack.pop()

    def exception_handler(self):
        if not self._handler_stack:
            raise
        kind, pc = self._handler_stack.pop()
        assert kind == 'except'
        self.pc = pc
        exctyp, excval, exctb = sys.exc_info()
        self.push(exctyp)
        self.push(excval)
        self.push(exctb)

    def push_handler(self, kind, pc):
        assert kind in ['except']
        self._handler_stack.append((kind, pc))

    def pop_handler(self):
        self._handler_stack.pop()

    def reraise(self, exctyp, excval, traceback):
        assert isinstance(excval, exctyp)
        raise excval.with_traceback(traceback)


def mangle(name):
    return '__CodeTrace_{0}_'.format(name)


def generate_state_machine(traces, srcbuf):
    vm = 'vm'
    fmt = '{vm} = {ctor}(locals(), globals())'
    srcbuf.println(fmt.format(vm=vm, ctor=mangle('VM_CTOR')))

    with srcbuf.indent_context('while True:'):
        with srcbuf.indent_context('try:'):
            generate_sm_switch(traces, srcbuf, vm)
        with srcbuf.indent_context('except:'):
            srcbuf.println('{vm}.exception_handler()'.format(vm=vm))


def generate_sm_switch(traces, srcbuf, vm):
    jump_table = {traces.entry_label: 0}

    for label in traces.labels:
        if label not in jump_table:
            jump_table[label] = len(jump_table)

    el = ''
    for label, state in traces.items():
        ifthen = '{el}if {vm}.pc == {cur}:'.format(el=el, vm=vm,
                                                   cur=jump_table[label])
        with srcbuf.indent_context(ifthen):
            instprinter = InstPrinter(traces=traces, vm=vm, srcbuf=srcbuf,
                                      state=state, jump_table=jump_table)
            instprinter.init_state()
            for inst in state.code:
                instprinter.process(inst)
            instprinter.fini_state()

        el = 'el'


def generate_sm_state(traces, srcbuf, pc, label, state):
    for inst in state.code:
        srcbuf.println(comment(inst))
        inst.opcode


class InstPrinter(object):
    def __init__(self, traces, vm, state, jump_table, srcbuf):
        self.traces = traces
        self.vm = vm
        self.state = state
        self.jump_table = jump_table
        self.srcbuf = srcbuf

    def init_state(self):
        """
        Invoked when a state starts
        """
        # unpack stack
        for val in self.state.incoming_stack:
            self._pop_stack(val)
        # setup exception handler
        hdlr = self.state.get_except_handler()
        if hdlr is not None:
            fmt = '{vm}.push_handler(kind={kind!r}, pc={pc})'
            kwargs = dict(vm=self.vm, pc=self.jump_table[hdlr],
                          kind='except')
            self.srcbuf.println(fmt.format(**kwargs))

    def fini_state(self):
        """
        Invoked when the state ends.

        Note:
        - Outgoing stack is prepared by the instruction that perform the jump
        """
        # remove exception handler
        hdlr = self.state.get_except_handler()
        if hdlr is not None:
            fmt = '{vm}.pop_handler()'
            self.srcbuf.println(fmt.format(vm=self.vm))

    def process(self, inst):
        if inst.opcode.startswith('.'):
            return
        fname = 'op_{0}'.format(inst.opcode)
        fn = getattr(self, fname)
        return fn(inst)

    def op_global(self, inst):
        self._store_local(dst=inst.out, src=inst.operands['name'])

    def op_const(self, inst):
        self._store_local(dst=inst.out, src=repr(inst.operands['value']))

    def op_load(self, inst):
        self._store_local(dst=inst.out,
                          src=self._load_local(inst.operands['src']))

    def op_store(self, inst):
        self._store_local(dst=inst.operands['dst'],
                          src=self._load_local(inst.operands['src']))

    def op_get_attr(self, inst):
        base = self._load_local(inst.operands['src'])
        src = '{base}.{attr}'.format(base=base,
                                     attr=inst.operands['attr'])
        self._store_local(dst=inst.out, src=src)

    def op_pop_block(self, inst):
        pass

    def op_push_block(self, inst):
        pass

    def op_prepare_args(self, inst):
        self._store_local(dst=inst.out, src='()')

    def op_append_args(self, inst):
        src = '{tup} + ({val},)'.format(
            tup=self._load_local(inst.operands['args']),
            val=self._load_local(inst.operands['value']))
        self._store_local(dst=inst.out, src=src)

    def op_call(self, inst):
        if 'kwargs' in inst.operands:
            raise NotImplementedError(inst)
        args = inst.operands['args']
        func = inst.operands['func']
        dst = inst.out
        src = '{func}(*{args})'.format(args=self._load_local(args),
                                       func=self._load_local(func))
        self._store_local(dst=dst, src=src)

    def op_get_iter(self, inst):
        iterobj = inst.operands['iter']
        dst = inst.out
        src = 'iter({0})'.format(self._load_local(iterobj))
        self._store_local(dst=dst, src=src)

    def op_for_iter(self, inst):
        iterobj = inst.operands['iter']
        with self.srcbuf.indent_context('try:'):
            src = 'next({0})'.format(self._load_local(iterobj))
            self._store_local(dst=inst.out, src=src)
        with self.srcbuf.indent_context('except StopIteration:'):
            self._jump_to_label(label=inst.operands['loop_exit'])
        with self.srcbuf.indent_context('else:'):
            self._jump_to_label(label=inst.operands['loop_entry'])

    def op_add(self, inst):
        self._binary_op('+', inst)

    def op_mul(self, inst):
        self._binary_op('*', inst)

    def op_cmp_lt(self, inst):
        self._binary_op('<', inst)

    def op_cmp_gt(self, inst):
        self._binary_op('>', inst)

    def op_cmp_exceptmatch(self, inst):
        kwargs = dict(dst=inst.out,
                      lhs=self._load_local(inst.operands['lhs']),
                      rhs=self._load_local(inst.operands['rhs']))
        src = 'issubclass({lhs}, {rhs})'.format(**kwargs)
        self._store_local(dst=inst.out, src=src)

    def op_ret(self, inst):
        value = self._load_local(inst.operands['value'])
        self.srcbuf.println('return {value}'.format(value=value))

    def op_reraise(self, inst):
        exctype = self._load_local(inst.operands['type'])
        excval = self._load_local(inst.operands['value'])
        traceback = self._load_local(inst.operands['traceback'])
        fmt = '{vm}.reraise({type}, {value}, {traceback})'
        self.srcbuf.println(fmt.format(vm=self.vm,
                                       type=exctype,
                                       value=excval,
                                       traceback=traceback))

    def op_jump(self, inst):
        self._jump_to_label(inst.operands['target'])

    def op_jump_if(self, inst):
        label_true = inst.operands['target_true']
        label_false = inst.operands['target_false']
        cond = self._load_local(inst.operands['cond'])
        with self.srcbuf.indent_context('if {cond}:'.format(cond=cond)):
            self._jump_to_label(label_true)

        with self.srcbuf.indent_context('else:'):
            self._jump_to_label(label_false)

    def _jump_to_label(self, label):
        target = self.jump_table[label]
        self.srcbuf.println('{vm}.pc = {target}'.format(vm=self.vm,
                                                        target=target))
        # pack stack
        for val in reversed(self.state.outgoing_stack(label)):
            self._push_stack(val)

    def _push_stack(self, val):
        self.srcbuf.println('{vm}.push({val})'.format(val=self._load_local(val),
                                                      vm=self.vm))

    def _pop_stack(self, val):
        self._store_local(dst=val, src='{vm}.pop()'.format(vm=self.vm))

    def _binary_op(self, op, inst):
        kwargs = dict(op=op, dst=inst.out,
                      lhs=self._load_local(inst.operands['lhs']),
                      rhs=self._load_local(inst.operands['rhs']))
        src = '{lhs} {op} {rhs}'.format(**kwargs)
        self._store_local(dst=inst.out, src=src)

    def _debug_print(self, text):
        self.srcbuf.println("print({0!r})".format(text))

    def _debug_eval(self, expr):
        self.srcbuf.println("print({0})".format(expr))

    def _store_local(self, dst, src):
        self.srcbuf.println('{vm}.store({src}, "{dst}")'.format(vm=self.vm,
                                                                dst=dst,
                                                                src=src))

    def _load_local(self, src):
        return '{vm}.load("{src}")'.format(vm=self.vm, src=src)


def comment(text):
    return "# {0}".format(text)


def format_def(name, argspec):
    return "def {name}({argspec}):".format(name=name, argspec=argspec)


def format_argspec(argspec):
    assert argspec.varargs is None
    assert argspec.keywords is None
    assert argspec.defaults is None
    return ', '.join(argspec.args)


class SourceBuffer(object):
    _indentchar = ' '
    _indentcount = 4

    def __init__(self):
        self._buf = []
        self._indentlevel = 0

    def println(self, text):
        assert self._indentlevel >= 0
        prefix = self._indentchar * (self._indentcount * self._indentlevel)
        self._buf.append(''.join([prefix, text]))

    @contextmanager
    def indent_context(self, text):
        self.println(text)
        self.indent()
        yield
        self.dedent()

    def indent(self):
        self._indentlevel += 1

    def dedent(self):
        assert self._indentlevel > 0
        self._indentlevel -= 1

    def show(self):
        return '\n'.join(self._buf)
