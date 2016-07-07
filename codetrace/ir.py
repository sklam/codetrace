
class Context(object):

    def __init__(self):
        self._counter = 0

    def make_use(self, val):
        return Use(self, val)

    def get_unique_id(self):
        ret = self._counter
        self._counter += 1
        return ret


class Block(object):

    def __init__(self, kind, **kwargs):
        self.kind = kind
        self.data = kwargs

    def __repr__(self):
        fmtdata = map("{0}={1}".format, self.data.items())
        return "Block({kind}, {data})".format(kind=self.kind, data=fmtdata)


class Use(object):

    def __init__(self, ctx, val):
        self._context = ctx
        self._name = str(ctx.get_unique_id())
        self._value = val

    def __repr__(self):
        return '%' + self._name

    def show(self):
        return "{0} = {1}".format(self, self._value)

    def equivalent(self, other):
        if type(self) == type(other):
            return self._value.equivalent(other._value)
        else:
            return False

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    def replace(self, value):
        self._value = value

    def get_uses(self):
        return self._value.get_uses()


class Inst(object):

    def show(self):
        return str(self)

    def equivalent(self, other):
        if type(self) == type(other):
            return self._equivalent(other)
        return False

    def _equivalent(self, other):
        return False


class Meta(Inst):

    def __init__(self, **kwargs):
        self.values = kwargs

    def __repr__(self):
        return "# {0}".format(', '.join("{0}={1}".format(k, v)
                                        for k, v in self.values.items()))

    def _equivalent(self, other):
        return self.values == other.values

    def get_uses(self):
        return set()


class Terminator(Inst):

    def outgoing_labels(self):
        return ()


class Jump(Terminator):

    def __init__(self, target, args):
        self.target = target
        self.args = tuple(args)

    def outgoing_labels(self):
        return (self.target,)

    def outgoing_stack(self, label):
        if isinstance(label, tuple):
            label = label[-1]

        if label == self.target:
            return self.args
        else:
            fmt = "invalid label: {0} expect {1}"
            raise ValueError(fmt.format(label, self.outgoing_labels()))

    def __repr__(self):
        fmt = "Jump(label {0} with {1})"
        return fmt.format(self.target, self.args)

    def _equivalent(self, other):
        return (self.target == other.target and
                equivalent_values(zip(reversed(self.args),
                                      reversed(other.args))))

    def replace_uses(self, mapping):
        return Jump(self.target, [mapping[x] for x in self.args])


class JumpIf(Terminator):

    def __init__(self, pred, then, orelse, then_args, else_args):
        self.pred = pred
        self.then = then
        self.orelse = orelse
        self.then_args = tuple(then_args)
        self.else_args = tuple(else_args)

    def __repr__(self):
        fmt = ("JumpIf({pred},\n\tlabel {then} with {then_args},\n\t"
               "label {orelse} with {else_args})")
        return fmt.format(pred=self.pred, then=self.then, orelse=self.orelse,
                          then_args=self.then_args, else_args=self.else_args)

    def outgoing_labels(self):
        return self.then, self.orelse

    def outgoing_stack(self, label):
        if isinstance(label, tuple):
            label = label[-1]

        if label == self.then:
            return self.then_args
        elif label == self.orelse:
            return self.else_args
        else:
            fmt = "invalid label {0} expect {1}"
            msg = fmt.format(label, self.outgoing_labels())
            raise ValueError(msg)

    def _equivalent(self, other):
        values = [(self.pred, other.pred)]
        values.extend(list(zip(reversed(self.then_args),
                               reversed(other.then_args))))
        values.extend(list(zip(reversed(self.else_args),
                               reversed(other.else_args))))
        return (self.then == other.then and
                self.orelse == other.orelse and
                equivalent_values(values))

    def replace_uses(self, mapping):
        return JumpIf(mapping[self.pred], self.then, self.orelse,
                      [mapping[x] for x in self.then_args],
                      [mapping[x] for x in self.else_args])

    def get_uses(self):
        return set([self.pred]) | set(self.then_args) | set(self.else_args)


class Ret(Terminator):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "Ret({value})".format(value=self.value)

    def _equivalent(self, other):
        return self.value.equivalent(other.value)

    def replace_uses(self, mapping):
        return Ret(mapping[self.value])

    def get_uses(self):
        return set([self.value])


class Value(Inst):
    pass


class Placeholder(Value):

    def __init__(self, name=''):
        self.name = name or "Placeholder({0})".format(hex(id(self)))

    def __repr__(self):
        return self.name

    def _equivalent(self, other):
        return True


class Op(Value):

    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def __repr__(self):
        return "Op({op}, {args})".format(op=self.op, args=self.args)

    def _equivalent(self, other):
        return (self.op == other.op and
                equivalent_values(zip(self.args, other.args)))

    def replace_uses(self, mapping):
        return Op(self.op, *(mapping[a] for a in self.args))

    def get_uses(self):
        return set(self.args)


class Const(Value):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "Const({value})".format(value=self.value)

    def _equivalent(self, other):
        return self.value == other.value

    def get_uses(self):
        return set([])


class LoadVar(Value):

    def __init__(self, name, scope):
        self.name = name
        self.scope = scope

    def __repr__(self):
        return "LoadVar({name}, {scope})".format(name=self.name,
                                                 scope=self.scope)

    def _equivalent(self, other):
        return (self.name == other.name and self.scope == other.scope)

    def get_uses(self):
        return set()


class StoreVar(Inst):

    def __init__(self, value, name, scope):
        self.value = value
        self.name = name
        self.scope = scope

    def __str__(self):
        return "StoreVar({value}, {name}, {scope})".format(value=self.value,
                                                           name=self.name,
                                                           scope=self.scope)

    def _equivalent(self, other):
        return (self.scope == other.scope and
                self.name == other.name and
                self.value.equivalent(other.value))

    def replace_uses(self, mapping):
        return StoreVar(value=mapping[self.value], name=self.name,
                        scope=self.scope)

    def get_uses(self):
        return set([self.value])


class Call(Value):

    def __init__(self, callee, args, kwargs):
        self.callee = callee
        self.args = tuple(args)
        self.kwargs = dict(kwargs)

    def __repr__(self):
        dct = dict(callee=self.callee, args=self.args, kwargs=self.kwargs)
        return "Call({callee}, {args}, {kwargs})".format(**dct)

    def _equivalent(self, other):
        return (self.callee.equivalent(other.callee) and
                list(self.kwargs.keys()) == list(other.kwargs.keys()) and
                equivalent_values(zip(self.args, other.args)) and
                equivalent_values(zip(self.kwargs.values(),
                                      other.kwargs.values())))

    def replace_uses(self, mapping):
        return Call(callee=mapping[self.callee],
                    args=map(mapping.__getitem__, self.args),
                    kwargs=dict((k, mapping[v])
                                for k, v in self.kwargs.items()))

    def get_uses(self):
        return set([self.callee]) | set(self.args) | set(self.kwargs.values())


def equivalent_values(pairs):
    return all(a.equivalent(b) for a, b in pairs)
