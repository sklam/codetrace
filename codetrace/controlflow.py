# Adapted from numba/controlflow.py
#
#
from __future__ import print_function, division, absolute_import

import collections
import functools
import sys
from itertools import permutations, product, combinations


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

    def _find_back_edges(self):
        """
        Find back edges.  Using DFS to detect cycles.
        """
        backedges = set()

        def dfs(stack):
            tos = stack[-1]
            succs = self._succs[tos]
            for other in succs:
                if other in stack:
                    backedges.add((tos, other))
                else:
                    dfs(stack + (other,))

        stack = tuple([self._entry_point])
        dfs(stack)
        self._back_edges = backedges

    def _find_topo_order(self):
        seen = set()
        temp = set()
        post_order = []

        def dfs(node):
            if node in temp or node in seen:
                return

            temp.add(node)
            succs = self._succs[node]
            for other in succs:
                dfs(other)
            temp.discard(node)
            seen.add(node)
            post_order.append(node)

        dfs(self._entry_point)
        self._topo_order = list(reversed(post_order))
        assert len(self._topo_order) == len(self._nodes)

    def _find_descendents(self):
        descs = collections.defaultdict(set)
        for node in reversed(self._topo_order):
            node_descs = descs[node]
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


# ------------------------------- Extended -------------------------------


class ExtCFGraph(CFGraph):

    def process(self):
        super(ExtCFGraph, self).process()
        self._build_dominator_tree()
        self._build_post_dominator_tree()
        self._build_region_tree()

    def graphviz(self, filename=None, view=True):
        from .gvutils import DigraphBuilder
        g = DigraphBuilder()
        for a, b in self.adjacency_list():
            g.edge(a, b)
        return g.render(filename=filename, view=view)

    def adjacency_list(self):
        return [(src, dst)
                for src, dests in self._succs.items()
                for dst in dests]

    def _build_dominator_tree(self):
        doms = self.dominators()
        cur = self.entry_point()
        tree = {cur: _build_dom_tree(cur, doms)}
        self._domtree = tree

    def _build_post_dominator_tree(self):
        pdoms = self.post_dominators()
        tree = {}
        for cur in self.exit_points():
            tree[cur] = _build_dom_tree(cur, pdoms)
        self._postdomtree = tree

    def dominator_tree(self):
        return self._domtree

    def post_dominator_tree(self):
        return self._postdomtree

    def region_tree(self):
        return self._regiontree

    def _build_region_tree(self):
        # Reference:
        # Johnson, Richard, David Pearson, and Keshav Pingali.
        # The program structure tree: Computing control regions in linear time.
        # ACM SigPlan Notices. Vol. 29. No. 6. ACM, 1994.
        # http://iss.ices.utexas.edu/Publications/Papers/PLDI1994.pdf

        # Find initial single-entry single-exit region
        doms = self.dominators()
        pdoms = self.post_dominators()

        all_nodes = frozenset(self.nodes())

        def dominates(x, y):
            return x in doms[y]

        def postdominates(x, y):
            return x in pdoms[y]

        regions = set()

        # rule 1: a dominates b
        # rule 2: b post-dominates a
        for a, b in product(all_nodes, repeat=2):
            if dominates(a, b) and postdominates(b, a):
                regions.add((a, b))

        # rule 3: cycle equivalence -- every cycle containing a contains b
        loops = self.loops().values()
        for loop in loops:
            body = loop.body
            for a, b in list(regions):
                intersect = set([a, b]) & body
                if len(intersect) == 1:
                    regions.discard((a, b))

        # node membership

        def contains(n, a, b):
            # node n is contained if for region (a, b) a dom n and b postdom n
            return dominates(a, n) and postdominates(b, n)

        membership = collections.defaultdict(set)
        for a, b in regions:
            for n in self.nodes():
                if contains(n, a, b):
                    membership[a, b].add(n)
        ranked = sorted(membership, key=lambda x: len(membership[x]))

        # filter non canonical regions
        canonical = set()

        def canonical_test(region, others):
            a, b = region
            # rule 1
            for c, d in others:
                if c is a:
                    # b must dominates d
                    if not dominates(b, d):
                        return False

                if d is b:
                    # a must postdominates c
                    if not postdominates(a, c):
                        return False

            return True

        canonical = set()
        # from small to big regions
        for region in ranked:
            if canonical_test(region, canonical):
                canonical.add(region)

        non_canonical = regions - canonical
        regions = set(canonical)

        for (a, b), (c, d) in combinations(canonical, r=2):
            overlap = membership[a, b] & membership[c, d]
            if overlap:
                # find a non-canonical
                repl = None
                if (a, d) in non_canonical:
                    repl = (a, d)
                elif (c, b) in non_canonical:
                    repl = (c, b)
                if repl is not None:
                    non_canonical.remove(repl)
                    regions.add(repl)
                    regions.remove((a, b))
                    regions.remove((c, d))

        # build up tree from bottom up
        def subregion_test(r, s):
            a, b = r.first, r.last
            c, d = s.first, s.last
            return contains(c, a, b) and contains(d, a, b)

        level = set()

        for a, b in [r for r in ranked if r in regions]:
            cur = Region(a, b)
            for sub in list(level):
                if subregion_test(cur, sub):
                    level.discard(sub)
                    cur.regions.add(sub)
            level.add(cur)

        if len(level) > 1:
            root = Region(self.entry_point(), None)
            root.regions |= level
            root.nodes |= all_nodes
        else:
            [root] = level

        # add non-canonical
        def add_non_canonical(root):
            inside = {}
            for (a, b) in non_canonical:
                top = Region(a, b)

                subregions = set()
                children = set()
                for sub in root.regions:
                    if subregion_test(top, sub):
                        subregions.add(sub)
                        children |= membership[sub.first, sub.last]
                # only include if new region is equivalent to subregions
                if children == membership[a, b]:
                    inside[top] = subregions

            ordered = sorted(inside, key=lambda k: len(inside[k]))
            while ordered:
                cur = ordered.pop()
                subregions = inside[cur]
                if (subregions & root.regions) != subregions:
                    continue
                root.regions -= subregions
                cur.regions |= subregions
                root.regions.add(cur)
                non_canonical.remove((cur.first, cur.last))

            for sub in root.regions:
                add_non_canonical(sub)

        add_non_canonical(root)

        # assign nodes
        def assign_nodes(tree):
            for sub in tree.regions:
                assign_nodes(sub)
            tree.nodes |= membership[tree.first, tree.last]
            for sub in tree.regions:
                tree.nodes -= set(sub.childnodes())

        assign_nodes(root)

        # unnecessary nesting
        if root.last is None and len(root.regions) == 1:  # and not root.nodes:
            [root] = root.regions

        self._regiontree = root

        # XXX unnecessary
        # verify
        def verify(tree, seen):
            assert not(seen & tree.nodes)
            seen |= tree.nodes
            for sub in tree:
                verify(sub, seen)

        seen = set()
        verify(root, seen)
        assert seen == self.nodes()


class Region(object):

    def __init__(self, first=None, last=None):
        self.first = first
        self.last = last
        self.nodes = set()
        self.regions = set()

    def __eq__(self, other):
        if isinstance(other, Region):
            return self.identity == other.identity
        return NotImplemented

    def __hash__(self):
        return hash(self.identity)

    def __iter__(self):
        return iter(self.regions)

    def __len__(self):
        return len(self.regions)

    def __repr__(self):
        return "<Region {self.first} {self.last} >".format(self=self)

    # def copy_nodes_from_subregions(self):
    #     for reg in self.regions:
    #         self.nodes |= reg.nodes

    @property
    def is_trivial(self):
        return self.first == self.last

    @property
    def is_extended(self):
        return self.last is None

    @property
    def identity(self):
        return self.first, self.last

    def show(self, indent=0, **kwargs):
        """show(indent=0, nodes=True)
        """
        prefix = ' ' * indent
        buf = []
        buf.append("{0}+ Region {1}:".format(prefix, self.identity))
        if kwargs.pop('nodes', True):
            buf.append("{0}    nodes: {1}".format(prefix, self.nodes))
        if self.regions:
            for reg in self.regions:
                inner = reg.show(indent=indent + 4, **kwargs)
                buf.append(inner)
        return '\n'.join(buf)

    def subregions(self):
        """
        Returns all sub-regions of this region recursing down the region tree
        """
        out = list(self.regions)
        for r in self:
            out.extend(r.subregions())
        return out

    def childnodes(self):
        """
        Returns all child-nodes of this region recursing down the region tree
        """
        out = list(self.nodes)
        for r in self.subregions():
            out.extend(r.nodes)
        return out


def _build_dom_tree(cur, doms):
    """
    Helper function to build dom and pdom trees recursively
    """
    # get all nodes that contains cur and is not cur
    candidates = set()
    for k, vl in doms.items():
        if cur in vl and k is not cur:
            candidates.add(k)
    # remove nodes that has a dom in candidates
    # this is finding the idoms
    excluded = set()
    for cand in tuple(candidates):
        cand_doms = set(doms[cand]) - set([cand])
        if cand_doms & candidates:
            excluded.add(cand)

    newroots = candidates - excluded
    # recursive build the tree
    out = {}
    for root in newroots:
        out[root] = _build_dom_tree(root, doms)
    return out
