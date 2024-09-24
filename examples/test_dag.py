import random
from pathlib import Path

from toyflow.dag import CycleError, Node, TopoSorter


def main():
    N = 10
    random.seed(42)
    no_loop = False
    while not no_loop:
        try:
            nodes = [Node(str(i)) for i in range(N)]
            for _ in range(N):
                nodes[random.randint(0, N - 1)].add_downstreams(nodes[random.randint(0, N - 1)])
            runner = TopoSorter.from_nodes(nodes)
            no_loop = True
        except CycleError:
            pass

    ref_order = []
    for _ in range(N):
        next_node = runner.get_next_node_candidates()[0]
        ref_order.append(next_node)
        runner.set_next_node(next_node)
    assert len(runner.get_remaining_nodes()) == 0
    assert runner.get_current_ordered_nodes() == tuple(ref_order)
    print(runner.get_current_ordered_nodes())
    # runner.graph.draw_graph(Path(__file__).parent / "dag_graph.png")


main()
