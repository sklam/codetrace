"""
A set of utilities regarding Python bytecode manipulation.
"""
from __future__ import absolute_import, print_function
import dis
from collections import namedtuple


def get_code_from_object(obj):
    code_obj = getattr(obj, '__code__', getattr(obj, 'func_code', None))
    if code_obj is None:
        raise TypeError("Cannot find code object in {0}".format(type(obj)))
    return code_obj


def parse_bytecode(code):
    """
    Parse bytecode from raw byte representation to provide structures.
    Returns a sequence of BytecodeInfo
    """
    code_bytes = code.co_code
    buf = []
    idx = 0

    lineinfos = dict(dis.findlinestarts(code))
    curline = 0
    while idx < len(code_bytes):
        curline = lineinfos.get(idx, curline)
        bc = BytecodeSpec(code_bytes[idx:])
        bcinfo = bc.pack(offset=idx, code=code, lineno=curline)
        buf.append(bcinfo)
        idx += len(bc)
    return buf


def find_jump_targets(bcinfos):
    """
    Find jump targets in the bytecode
    """
    targets = set()
    for inst in bcinfos:
        if inst.is_jabs or inst.is_jrel:
            targets.add(get_jump_target(inst))
        if inst.is_cbr:
            targets.add(inst.next)
    return targets


def get_jump_target(inst):
    if not (inst.is_jrel or inst.is_jabs):
        raise ValueError("not a jump instruction")
    res = inst.args[0]
    if inst.is_jrel:
        res += inst.next
    return res


BytecodeInfo = namedtuple("BytecodeInfo", ["opname",
                                           "offset",
                                           "args",
                                           "is_jabs",
                                           "is_jrel",
                                           "is_cbr",
                                           "next",
                                           "lineno"])


class BytecodeSpec(object):
    registry = {}
    # override the following in subclass
    nbytes = 0
    stack_push = 0
    stack_pop = 0
    is_conditional_jump = False

    def __new__(cls, prefix_bytes):
        prefix = prefix_bytes[0]
        return object.__new__(cls.registry[dis.opname[prefix]])

    def __init__(self, prefix_bytes):
        assert self.nbytes > 0
        self.raw_bytes = prefix_bytes[:self.nbytes]
        self.arg_bytes = self.raw_bytes[1:]
        self.opcode = self.raw_bytes[0]

    def __len__(self):
        return self.nbytes

    def pack(self, offset, code, lineno):
        return BytecodeInfo(opname=type(self).__name__,
                            offset=offset,
                            args=self.get_arguments(code),
                            is_jabs=self.is_jabs,
                            is_jrel=self.is_jrel,
                            is_cbr=self.is_conditional_jump,
                            next=offset + self.nbytes,
                            lineno=lineno)

    @classmethod
    def define(cls, impcls):
        assert issubclass(impcls, BytecodeSpec)
        cls.registry[impcls.__name__] = impcls
        return impcls

    def get_arguments(self, code):
        """
        Override this method in subclass
        """
        return ()

    @classmethod
    def as_int(cls, bytes):
        val = 0
        for v in reversed(bytes):
            val <<= 8
            val |= v
        return val

    @property
    def is_jabs(self):
        """
        Is a absolute jump?
        """
        return self.opcode in dis.hasjabs

    @property
    def is_jrel(self):
        """
        Is a relative jump?
        """
        return self.opcode in dis.hasjrel


class LoadStore(BytecodeSpec):
    nbytes = 3

    def get_arguments(self, code):
        var = self.as_int(self.raw_bytes[1:])
        return (code.co_varnames[var],)


@BytecodeSpec.define
class POP_TOP(BytecodeSpec):
    nbytes = 1


@BytecodeSpec.define
class DUP_TOP(BytecodeSpec):
    nbytes = 1


@BytecodeSpec.define
class LOAD_CONST(BytecodeSpec):
    nbytes = 3

    def get_arguments(self, code):
        which = self.as_int(self.raw_bytes[1:])
        return (code.co_consts[which],)


@BytecodeSpec.define
class LOAD_GLOBAL(BytecodeSpec):
    nbytes = 3

    def get_arguments(self, code):
        which = self.as_int(self.raw_bytes[1:])
        return (code.co_names[which],)


@BytecodeSpec.define
class LOAD_FAST(LoadStore):
    pass


@BytecodeSpec.define
class STORE_FAST(LoadStore):
    pass


@BytecodeSpec.define
class LOAD_ATTR(BytecodeSpec):
    nbytes = 3

    def get_arguments(self, code):
        which = self.as_int(self.raw_bytes[1:])
        return (code.co_names[which],)


@BytecodeSpec.define
class COMPARE_OP(BytecodeSpec):
    nbytes = 3

    def get_arguments(self, code):
        return (dis.cmp_op[self.as_int(self.arg_bytes)],)


class HasJumpArg(BytecodeSpec):
    nbytes = 3

    def get_arguments(self, code):
        return (self.as_int(self.arg_bytes),)


class ConditionalJump(HasJumpArg):
    is_conditional_jump = True


@BytecodeSpec.define
class POP_JUMP_IF_FALSE(ConditionalJump):
    pass


@BytecodeSpec.define
class JUMP_IF_TRUE_OR_POP(ConditionalJump):
    pass


@BytecodeSpec.define
class JUMP_FORWARD(HasJumpArg):
    pass


@BytecodeSpec.define
class JUMP_ABSOLUTE(HasJumpArg):
    pass


@BytecodeSpec.define
class SETUP_LOOP(HasJumpArg):
    pass


@BytecodeSpec.define
class POP_BLOCK(BytecodeSpec):
    nbytes = 1


class BinaryOp(BytecodeSpec):
    nbytes = 1


@BytecodeSpec.define
class BINARY_ADD(BinaryOp):
    pass


@BytecodeSpec.define
class BINARY_SUBTRACT(BinaryOp):
    pass


@BytecodeSpec.define
class BINARY_MULTIPLY(BinaryOp):
    pass


@BytecodeSpec.define
class RETURN_VALUE(BytecodeSpec):
    nbytes = 1


@BytecodeSpec.define
class CALL_FUNCTION(BytecodeSpec):
    nbytes = 3

    def get_arguments(self, code):
        word = self.as_int(self.arg_bytes)
        nkey = (word >> 8) & 0xff
        npos = word & 0xff
        return (npos, nkey)


@BytecodeSpec.define
class GET_ITER(BytecodeSpec):
    nbytes = 1


@BytecodeSpec.define
class FOR_ITER(ConditionalJump):
    pass


@BytecodeSpec.define
class SETUP_EXCEPT(HasJumpArg):
    pass


@BytecodeSpec.define
class POP_EXCEPT(BytecodeSpec):
    nbytes = 1


@BytecodeSpec.define
class END_FINALLY(BytecodeSpec):
    nbytes = 1


@BytecodeSpec.define
class SETUP_FINALLY(HasJumpArg):
    pass


@BytecodeSpec.define
class SETUP_WITH(HasJumpArg):
    pass


@BytecodeSpec.define
class WITH_CLEANUP(BytecodeSpec):
    nbytes = 1
