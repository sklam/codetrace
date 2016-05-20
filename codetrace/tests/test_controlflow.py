import unittest

from codetrace.controlflow import ExtCFGraph


class TestControlFlow(unittest.TestCase):

    def test_non_natural_loop(self):
        data = """
                0 1
                2 3
                4 3
                5 6
                7 8
                9 5
                8 4
                0 7
                7 9
                10 11
                9 12
                2 13
                5 2
                14 5
                14 12
                13 14
                4 13
                3 4
                10 0
                """
        cfg = ExtCFGraph()

        for line in data.strip().splitlines():
            a, b = map(int, line.strip().split())
            cfg.add_node(a)
            cfg.add_node(b)
            cfg.add_edge(a, b)
        cfg.set_entry_point(10)
        cfg.process()
        # cfg.dump()
        # cfg.graphviz()
        loops = cfg.loops()
        self.assertTrue(loops, "no loops were detected")
        self.assertEqual(loops[3].entries, set([2]))
        self.assertEqual(loops[4].entries, set([8]))
        self.assertEqual(loops[5].entries, set([9]))
        self.assertEqual(loops[13].entries, set([4]))


if __name__ == '__main__':
    unittest.main()
