from dataclasses import dataclass

@dataclass
class BeadNode:
    bead_id: str
    side: int | None   # +1, -1, or None

@dataclass
class PolymerGraph:
    nodes: list[BeadNode]
    edges: list[tuple[int, int]]

    @classmethod
    def linear(cls, nodes):
        edges = [(i, i+1) for i in range(len(nodes)-1)]
        return cls(nodes, edges)

