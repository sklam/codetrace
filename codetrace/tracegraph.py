from collections import defaultdict, MutableMapping

from .state import MalformStateError
from .ir import JumpIf, Jump


class TraceGraph(MutableMapping):
    first_key = None, 0

    def __init__(self):
        self._states = {}

    def __contains__(self, key):
        return key in self._states

    def __getitem__(self, key):
        return self._states[key]

    def __setitem__(self, key, value):
        self._states[key] = value

    def __delitem__(self, key):
        # intentionally disabled
        raise NotImplementedError('deletion is not supported')

    def __iter__(self):
        return iter(self._states)

    def __len__(self):
        return len(self._states)

    @property
    def first(self):
        return self[self.first_key]

    def verify(self):
        # verify each state
        for key, st in self._states.items():
            st.verify()

    def verify_stack_agreement(self):
        for key, st in self._states.items():
            for label in st.outgoing_labels():
                target = self._states[label]
                if not st.branch_stack_agree(target):
                    msg = 'at branch {0}, stack disagree'.format(label)
                    raise MalformStateError(msg)

    def simplify(self):
        self.merge_equivalent_states()
        self.combine_straight_line_jumps()

    def combine_straight_line_jumps(self):
        """
        Combine states that are unconditionally jumping to the target that has
        no other precedecessors.
        """
        # count precedecessors
        predct = defaultdict(int)
        for st in self._states.values():
            for edge in st.outgoing_labels():
                target = self._states[edge]
                predct[target] += 1

        # combine
        potential_targets = set([st for st, ct in predct.items() if ct == 1])
        candidates = set(filter(lambda st: isinstance(st.terminator, Jump),
                                self._states.values()))
        while candidates:
            st = candidates.pop()
            other_key = st.outgoing_labels()[0]
            other = self._states[other_key]
            if other in potential_targets:
                st.combine(other)
                # remove other state
                del self._states[other_key]

                # fix other edges
                for other_edge in other.outgoing_labels():
                    self._states[(st, other_edge[1])] = self._states[other_edge]
                    del self._states[other_edge]

                # drop other from candidiates
                if other in candidates:
                    candidates.add(st)
                    print("discard", other, other.offset)
                    candidates.discard(other)

    def merge_equivalent_states(self):
        # find candidates: states with the same offset
        candidates = defaultdict(set)
        for (origin, offset), st in self._states.items():
            candidates[offset].add(st)
        # prune candidates set that are less than two elements
        premerge = [list(vs) for k, vs in candidates.items() if len(vs) >= 2]
        # merge
        mapping = {}
        for group in premerge:
            while len(group) >= 2:
                head = group.pop()
                failed = []
                for other in group:
                    if head.can_merge(other):
                        mapping[other] = head
                    else:
                        failed.append(other)
                group = failed
        # apply replacement
        self._states = dict([(key, mapping.get(st, st))
                             for key, st in self._states.items()])

    def find_mergeable(self, edge, state):
        # find mergeable states
        st, pc = edge
        for (st, pc), other in self.items():
            if (_maybegetpc(st), pc) == (_maybegetpc(edge[0]), edge[1]):
                if state.can_merge(other):
                    return other

    def dump(self):
        for key, st in self._states.items():
            print(key)
            print(st.show())

    def graphviz(self, filename='codetrace_dot.dot', view=True):
        from graphviz import Digraph

        def fix_line_break(text):
            return ''.join(ln + '\\l' for ln in text.splitlines())

        styling = dict(fontname='courier', fontsize='9pt')
        g = Digraph(node_attr=styling, edge_attr=styling)

        for key, st in sorted(self._states.items(), key=lambda x: x[0][1]):
            g.node(str(id(st)), label=fix_line_break(st.show()), shape='rect')

        for st in set(self._states.values()):
            if st._instlist:
                for label in st.outgoing_labels():
                    opts = {}
                    if label not in self._states:
                        # missing
                        dest_node = 'missing ' + str(label)
                    else:
                        dest_node = str(id(self._states[label]))
                        if not st.branch_stack_agree(self._states[label]):
                            opts['color'] = 'red'
                    desc = '<unknown>'
                    term = st.terminator
                    if isinstance(term, JumpIf):
                        desc = 'then' if label[-1] == term.then else 'else'
                    elif isinstance(term, Jump):
                        desc = 'jump'

                    g.edge(str(id(st)), dest_node,
                           label=desc, **opts)

        g.render(filename, view=view)


def _maybegetpc(x):
    return None if x is None else x.pc
