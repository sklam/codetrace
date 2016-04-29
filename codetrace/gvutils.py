import graphviz


def _fix_line_break(text):
    return ''.join(ln + '\\l' for ln in text.splitlines())


class DigraphBuilder(object):

    def __init__(self, name=None, attrs=[]):
        styling = dict(fontname='courier', fontsize='9pt')
        self.graph = graphviz.Digraph(name, node_attr=styling,
                                      edge_attr=styling)
        if attrs:
            body = self.graph.body
            for a in attrs:
                body.append(str(a))

    def node(self, node, label=None, **kwargs):
        if label is not None:
            kwargs['label'] = _fix_line_break(label)
        self.graph.node(str(node), **kwargs)

    def edge(self, src, dst, label=None, **kwargs):
        self.graph.edge(str(src), str(dst), label=label, **kwargs)

    def subgraph(self, subgraph):
        self.graph.subgraph(subgraph.graph)

    def render(self, filename, view):
        self.graph.render(filename, view=view)
        return self.graph
