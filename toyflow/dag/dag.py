import os
from pathlib import Path
from typing import List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from toyflow.utils import logger

__all__ = [
    "Node",
    "CycleError",
    "Graph",
    "TopoSorter",
]


class Node:
    def __init__(self, name) -> None:
        self.name = name
        self._upstreams = set()
        self._downstreams = set()

    def __repr__(self) -> str:
        return f"{self.name}"

    def add_upstreams(self, nodes: Union["Node", List["Node"]]):
        if isinstance(nodes, Node):
            nodes = [nodes]
        self._upstreams.update(nodes)

    def add_downstreams(self, nodes: Union["Node", List["Node"]]):
        if isinstance(nodes, Node):
            nodes = [nodes]
        self._downstreams.update(nodes)

    @property
    def upstreams(self):
        return self._upstreams

    @property
    def downstreams(self):
        return self._downstreams


class CycleError(ValueError):
    pass


class Graph:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    @property
    def num_nodes(self):
        return len(self.nodes)

    def resolve(self, nodes: Optional[List[Node]] = None):
        """
        Returns upstreams and downstreams dicts.
        For each dict item, the key is a node,
        and the value is a set of upstream / downstream nodes.
        """
        nodes = self.nodes if nodes is None else nodes
        upstreams = {node: set() for node in nodes}
        downstreams = {node: set() for node in nodes}

        for node in nodes:
            upstreams[node].update(node.upstreams)
            downstreams[node].update(node.downstreams)
            for upstream in node.upstreams:
                downstreams[upstream].add(node)
            for downstream in node.downstreams:
                upstreams[downstream].add(node)

        return upstreams, downstreams

    def check_validity(self):
        """
        Returns True if the graph is a DAG. Otherwise returns False.
        """
        upstreams, downstreams = self.resolve(self.nodes)
        in_degree = {node: len(upstreams[node]) for node in upstreams}

        root_nodes = []
        for node in in_degree:
            if in_degree[node] == 0:
                root_nodes.append(node)

        while root_nodes:
            node = root_nodes.pop()
            for downstream in downstreams[node]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    root_nodes.append(downstream)

        return sum(in_degree.values()) == 0

    def draw_graph(self, file_path: Union[str, Path, os.PathLike]):
        """
        Draws the graph using matplotlib.
        """
        _, downstreams = self.resolve(self.nodes)
        edges = []
        for node in downstreams:
            for down_node in downstreams[node]:
                edges.append((node, down_node))

        matplotlib.use("agg")  # fix for wsl
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        # nx.write_graphml_xml(G, file_path)
        path = Path(file_path).resolve().as_posix()
        plt.savefig(path)
        logger.info("Graph is saved at %s.", file_path)


class TopoSorter:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        if not self.graph.check_validity():
            raise CycleError("The graph contains loop.")

        self._upstreams, self._downstreams = self.graph.resolve()
        self._in_degree = {node: len(self._upstreams[node]) for node in self._upstreams}
        self._root_nodes = []
        for node in self._in_degree:
            if self._in_degree[node] == 0:
                self._root_nodes.append(node)
        self._order: List[Node] = []

    @classmethod
    def from_graph(cls, graph: Graph):
        return cls(graph=graph)

    @classmethod
    def from_nodes(cls, nodes: List[Node]):
        return cls(Graph(nodes))

    def get_next_node_candidates(self) -> tuple[Node, ...]:
        return tuple(self._root_nodes)

    def set_next_node(self, node: Node):
        if node not in self._root_nodes:
            raise ValueError(f"This node {node} is not available now.")
        self._root_nodes.remove(node)
        self._order.append(node)
        new_root_nodes = []
        for downstream in self._downstreams[node]:
            self._in_degree[downstream] -= 1
            if self._in_degree[downstream] == 0:
                new_root_nodes.append(downstream)
                self._root_nodes.append(downstream)
        return set(new_root_nodes)

    def get_current_ordered_nodes(self):
        return tuple(self._order)

    def get_remaining_nodes(self):
        return tuple(node for node in self._upstreams if node not in self._order)
