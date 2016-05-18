from __future__ import absolute_import, print_function

from pprint import pprint
from . import controlflow


class CFA(object):

    def __init__(self, tracegraph):
        self._cfg = controlflow.ExtCFGraph()
        for state in tracegraph.values():
            self._cfg.add_node(state)

        for state in tracegraph.values():
            for label in state.outgoing_labels():
                target = tracegraph[label]
                self._cfg.add_edge(state, target)

        self._cfg.set_entry_point(tracegraph[tracegraph.first_key])
        self._cfg.process()
        # self._cfg.dump()

    @property
    def cfg(self):
        return self._cfg

    def region_tree(self):
        return self._cfg.region_tree()

    def gv_dominator_tree(self, filename='domtree.gv', view=True):
        from .gvutils import DigraphBuilder

        g = DigraphBuilder()
        dt = self._cfg.dominator_tree()
        dot_dom_tree(g, dt)
        return g.render(filename, view=view)

    def gv_post_dominator_tree(self, filename='postdomtree.gv', view=True):
        from .gvutils import DigraphBuilder

        g = DigraphBuilder()
        dt = self._cfg.post_dominator_tree()
        dot_dom_tree(g, dt)
        return g.render(filename, view=view)

    def gv_region_tree(self, tracegraph, filename='regiontree.gv', **kwargs):
        return tracegraph.graphviz(postfn=self._gv_region_tree_postfn,
                                   filename=filename, **kwargs)

    def _gv_region_tree_postfn(self, g):
        """
        For use as ``postfn`` in ``TraceGraph.graphviz``.
        """
        clusters = []

        def draw_region_tree(parent, regiontree, drawn=set()):
            from .gvutils import DigraphBuilder

            color = 'blue'
            if regiontree.is_trivial:
                color = 'green'
            if regiontree.is_extended:
                color = 'red'

            subgraph = DigraphBuilder('cluster_{0}'.format(len(clusters)),
                                      attrs=['color=' + color, 'style=dashed'])
            clusters.append(subgraph)

            # Draw sub region tree depth first and recursively
            for r in regiontree:
                draw_region_tree(subgraph, r, drawn)

            # Draw nodes in this region if they are not drawn yet
            for st in regiontree.nodes:
                assert st not in drawn, 'malform regiontree {0}'.format(st)
                subgraph.node(id(st))
                drawn.add(st)

            # Add cluster to parent graph
            parent.subgraph(subgraph)

        entry = self.cfg.entry_point()
        g.edge('entry', id(entry))

        drawn = set()
        draw_region_tree(g, self.region_tree(), drawn)

        for exit in self.cfg.exit_points():
            g.edge(id(exit), 'exit')

    def region_local_cfg(self, tracegraph, region):
        cfg = controlflow.ExtCFGraph()

        # handle nodes
        for s in region.nodes:
            cfg.add_node(s)
        cfg.add_node(None)

        # handle subregions
        submap = {}

        for r in region.regions:
            submap[r.first] = r
            for s in r.childnodes():
                submap[s] = r

            cfg.add_node(r)

        def sub(x):
            y = submap.get(x, x)
            if y is x and x not in region.nodes:
                return None
            return y

        entry = sub(region.first)
        assert entry is not None
        cfg.set_entry_point(entry)

        for s in region.nodes:
            for label in s.outgoing_labels():
                target = tracegraph[label]
                cfg.add_edge(s, sub(target))

        for r in region.regions:
            tail = r.last
            for label in tail.outgoing_labels():
                target = sub(tracegraph[label])
                # avoid self reference by regions
                if target is not r:
                    cfg.add_edge(r, target)

        cfg.process()

        return cfg


def dot_dom_tree(g, node):
    for k, vl in node.items():
        for v in vl:
            g.edge(k, v)
        dot_dom_tree(g, vl)
