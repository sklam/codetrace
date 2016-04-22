# Adapted from numba/controlflow.py
#
#
from __future__ import print_function, division, absolute_import

import collections
import functools
import sys


class CFBlock(object):

    def __init__(self, offset):
        self.offset = offset
        self.body = []
        # A map of jumps to outgoing blocks (successors):
        #   { offset of outgoing block -> number of stack pops }
        self.outgoing_jumps = {}
        # A map of jumps to incoming blocks (predecessors):
        #   { offset of incoming block -> number of stack pops }
        self.incoming_jumps = {}
        self.terminating = False

    def __repr__(self):
        args = self.offset, sorted(
            self.outgoing_jumps), sorted(self.incoming_jumps)
        return "block(offset:%d, outgoing: %s, incoming: %s)" % args

    def __iter__(self):
        return iter(self.body)


class Loop(
        collections.namedtuple("Loop", ("entries", "exits", "header", "body"))):
    """
    A control flow loop, as detected by a CFGraph object.
    """

    __slots__ = ()

    # The loop header is enough to detect that two loops are really
    # the same, assuming they belong to the same graph.
    # (note: in practice, only one loop instance is created per graph
    #  loop, so identity would be fine)

    def __eq__(self, other):
        return isinstance(other, Loop) and other.header == self.header

    def __hash__(self):
        return hash(self.header)


class CFGraph(object):
    """
    Generic (almost) implementation of a Control Flow Graph.
    """

    def __init__(self):
        self._nodes = set()
        self._preds = collections.defaultdict(set)
        self._succs = collections.defaultdict(set)
        self._edge_data = {}
        self._entry_point = None

    def add_node(self, node):
        """
        Add *node* to the graph.  This is necessary before adding any
        edges from/to the node.  *node* can be any hashable object.
        """
        self._nodes.add(node)

    def add_edge(self, src, dest, data=None):
        """
        Add an edge from node *src* to node *dest*, with optional
        per-edge *data*.
        If such an edge already exists, it is replaced (duplicate edges
        are not possible).
        """
        assert src in self._nodes
        assert dest in self._nodes
        self._add_edge(src, dest, data)

    def successors(self, src):
        """
        Yield (node, data) pairs representing the successors of node *src*.
        (*data* will be None if no data was specified when adding the edge)
        """
        for dest in self._succs[src]:
            yield dest, self._edge_data[src, dest]

    def predecessors(self, dest):
        """
        Yield (node, data) pairs representing the predecessors of node *dest*.
        (*data* will be None if no data was specified when adding the edge)
        """
        for src in self._preds[dest]:
            yield src, self._edge_data[src, dest]

    def set_entry_point(self, node):
        """
        Set the entry point of the graph to *node*.
        """
        assert node in self._nodes
        self._entry_point = node

    def process(self):
        """
        Compute various properties of the control flow graph.  The graph
        must have been fully populated, and its entry point specified.
        """
        if self._entry_point is None:
            raise RuntimeError("no entry point defined!")
        self._eliminate_dead_blocks()
        self._find_exit_points()
        self._find_dominators()
        self._find_back_edges()
        self._find_topo_order()
        self._find_descendents()
        self._find_loops()
        self._find_post_dominators()

    def dominators(self):
        """
        Return a dictionary of {node -> set(nodes)} mapping each node to
        the nodes dominating it.

        A node D dominates a node N when any path leading to N must go through D.
        """
        return self._doms

    def post_dominators(self):
        """
        Return a dictionary of {node -> set(nodes)} mapping each node to
        the nodes post-dominating it.

        A node P post-dominates a node N when any path starting from N must go
        through P.
        """
        return self._post_doms

    def descendents(self, node):
        """
        Return the set of descendents of the given *node*, in topological
        order (ignoring back edges).
        """
        return self._descs[node]

    def entry_point(self):
        """
        Return the entry point node.
        """
        assert self._entry_point is not None
        return self._entry_point

    def exit_points(self):
        """
        Return the computed set of exit nodes (may be empty).
        """
        return self._exit_points

    def backbone(self):
        """
        Return the set of nodes constituting the graph's backbone.
        (i.e. the nodes that every path starting from the entry point
         must go through).  By construction, it is non-empty: it contains
         at least the entry point.
        """
        return self._post_doms[self._entry_point]

    def loops(self):
        """
        Return a dictionary of {node -> loop} mapping each loop header
        to the loop (a Loop instance) starting with it.
        """
        return self._loops

    def in_loops(self, node):
        """
        Return the list of Loop objects the *node* belongs to,
        from innermost to outermost.
        """
        return [self._loops[x] for x in self._in_loops[node]]

    def dead_nodes(self):
        """
        Return the set of dead nodes (eliminated from the graph).
        """
        return self._dead_nodes

    def nodes(self):
        """
        Return the set of live nodes.
        """
        return self._nodes

    def topo_order(self):
        """
        Return the sequence of nodes in topological order (ignoring back
        edges).
        """
        return self._topo_order

    def topo_sort(self, nodes, reverse=False):
        """
        Iterate over the *nodes* in topological order (ignoring back edges).
        The sort isn't guaranteed to be stable.
        """
        nodes = set(nodes)
        it = self._topo_order
        if reverse:
            it = reversed(it)
        for n in it:
            if n in nodes:
                yield n

    def dump(self, file=None):
        """
        Dump extensive debug information.
        """
        import pprint
        file = file or sys.stdout
        if 1:
            print("CFG adjacency lists:", file=file)
            self._dump_adj_lists(file)
        print("CFG dominators:", file=file)
        pprint.pprint(self._doms, stream=file)
        print("CFG post-dominators:", file=file)
        pprint.pprint(self._post_doms, stream=file)
        print("CFG back edges:", sorted(self._back_edges), file=file)
        print("CFG loops:", file=file)
        pprint.pprint(self._loops, stream=file)
        print("CFG node-to-loops:", file=file)
        pprint.pprint(self._in_loops, stream=file)

    # Internal APIs

    def _add_edge(self, from_, to, data=None):
        # This internal version allows adding edges to/from unregistered
        # (ghost) nodes.
        self._preds[to].add(from_)
        self._succs[from_].add(to)
        self._edge_data[from_, to] = data

    def _remove_node_edges(self, node):
        for succ in self._succs.pop(node, ()):
            self._preds[succ].remove(node)
            del self._edge_data[node, succ]
        for pred in self._preds.pop(node, ()):
            self._succs[pred].remove(node)
            del self._edge_data[pred, node]

    def _dfs(self, entries=None):
        if entries is None:
            entries = (self._entry_point,)
        seen = set()
        stack = list(entries)
        while stack:
            node = stack.pop()
            if node not in seen:
                yield node
                seen.add(node)
                for succ in self._succs[node]:
                    stack.append(succ)

    def _eliminate_dead_blocks(self):
        """
        Eliminate all blocks not reachable from the entry point, and
        stash them into self._dead_nodes.
        """
        live = set()
        for node in self._dfs():
            live.add(node)
        self._dead_nodes = self._nodes - live
        self._nodes = live
        # Remove all edges leading from dead nodes
        for dead in self._dead_nodes:
            self._remove_node_edges(dead)

    def _find_exit_points(self):
        """
        Compute the graph's exit points.
        """
        exit_points = set()
        for n in self._nodes:
            if not self._succs.get(n):
                exit_points.add(n)
        self._exit_points = exit_points

    def _find_dominators_internal(self, post=False):
        # See theoretical description in
        # http://en.wikipedia.org/wiki/Dominator_%28graph_theory%29
        # The algorithm implemented here uses a todo-list as described
        # in http://pages.cs.wisc.edu/~fischer/cs701.f08/finding.loops.html
        if post:
            entries = set(self._exit_points)
            preds_table = self._succs
            succs_table = self._preds
        else:
            entries = set([self._entry_point])
            preds_table = self._preds
            succs_table = self._succs

        if not entries:
            raise RuntimeError("no entry points: dominator algorithm "
                               "cannot be seeded")

        doms = {}
        for e in entries:
            doms[e] = set([e])

        todo = []
        for n in self._nodes:
            if n not in entries:
                doms[n] = set(self._nodes)
                todo.append(n)

        while todo:
            n = todo.pop()
            if n in entries:
                continue
            new_doms = set([n])
            preds = preds_table[n]
            if preds:
                new_doms |= functools.reduce(set.intersection,
                                             [doms[p] for p in preds])
            if new_doms != doms[n]:
                assert len(new_doms) < len(doms[n])
                doms[n] = new_doms
                todo.extend(succs_table[n])
        return doms

    def _find_dominators(self):
        self._doms = self._find_dominators_internal(post=False)

    def _find_post_dominators(self):
        # To handle infinite loops correctly, we need to add a dummy
        # exit point, and link members of infinite loops to it.
        dummy_exit = object()
        self._exit_points.add(dummy_exit)
        for loop in self._loops.values():
            if not loop.exits:
                for b in loop.body:
                    self._add_edge(b, dummy_exit)
        self._post_doms = self._find_dominators_internal(post=True)
        # Fix the _post_doms table to make no reference to the dummy exit
        del self._post_doms[dummy_exit]
        for doms in self._post_doms.values():
            doms.discard(dummy_exit)
        self._remove_node_edges(dummy_exit)
        self._exit_points.remove(dummy_exit)

    # Finding loops and back edges: see
    # http://pages.cs.wisc.edu/~fischer/cs701.f08/finding.loops.html

    def _find_back_edges(self):
        """
        Find back edges.  An edge (src, dest) is a back edge if and
        only if *dest* dominates *src*.
        """
        back_edges = set()
        for src, succs in self._succs.items():
            back = self._doms[src] & succs
            # In CPython bytecode, at most one back edge can flow from a
            # given block.
            assert len(back) <= 1
            back_edges.update((src, dest) for dest in back)
        self._back_edges = back_edges

    def _find_topo_order(self):
        succs = self._succs
        back_edges = self._back_edges
        post_order = []
        seen = set()

        def _dfs_rec(node):
            if node not in seen:
                seen.add(node)
                for dest in succs[node]:
                    if (node, dest) not in back_edges:
                        _dfs_rec(dest)
                post_order.append(node)

        _dfs_rec(self._entry_point)
        post_order.reverse()
        self._topo_order = post_order

    def _find_descendents(self):
        descs = {}
        for node in reversed(self._topo_order):
            descs[node] = node_descs = set()
            for succ in self._succs[node]:
                if (node, succ) not in self._back_edges:
                    node_descs.add(succ)
                    node_descs.update(descs[succ])
        self._descs = descs

    def _find_loops(self):
        """
        Find the loops defined by the graph's back edges.
        """
        bodies = {}
        for src, dest in self._back_edges:
            # The destination of the back edge is the loop header
            header = dest
            # Build up the loop body from the back edge's source node,
            # up to the source header.
            body = set([header])
            queue = [src]
            while queue:
                n = queue.pop()
                if n not in body:
                    body.add(n)
                    queue.extend(self._preds[n])
            # There can be several back edges to a given loop header;
            # if so, merge the resulting body fragments.
            if header in bodies:
                bodies[header].update(body)
            else:
                bodies[header] = body

        # Create a Loop object for each header.
        loops = {}
        for header, body in bodies.items():
            entries = set()
            exits = set()
            for n in body:
                entries.update(self._preds[n] - body)
                exits.update(self._succs[n] - body)
            loop = Loop(header=header, body=body, entries=entries, exits=exits)
            loops[header] = loop
        self._loops = loops

        # Compute the loops to which each node belongs.
        in_loops = dict((n, []) for n in self._nodes)
        # Sort loops from longest to shortest
        # This ensures that outer loops will come before inner loops
        for loop in sorted(loops.values(), key=lambda loop: len(loop.body)):
            for n in loop.body:
                in_loops[n].append(loop.header)
        self._in_loops = in_loops

    def _dump_adj_lists(self, file):
        adj_lists = dict((src, list(dests))
                         for src, dests in self._succs.items())
        import pprint
        pprint.pprint(adj_lists, stream=file)