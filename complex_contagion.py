import random
from enum import Enum

import networkx as nx
import numpy as np
import scipy.io


class BehaviorState(Enum):
    """Agent states in the complex contagion model."""

    BASELINE = "B"  # Blue
    ADOPTED = "A"   # Aquamarine/cyan


def get_graph_nineteen_four_from_ncm_book():
    """Create the graph from Figure 19.4 of Networks, Crowds, and Markets."""
    G = nx.Graph()
    G.add_nodes_from(range(1, 18))  # Nodes 1-17 (edges reference 16, 17)
    G.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 6),
            (4, 5),
            (4, 6),
            (4, 7),
            (5, 7),
            (5, 8),
            (6, 9),
            (7, 8),
            (7, 9),
            (7, 10),
        ]
    )
    G.add_edges_from(
        [
            (8, 10),
            (8, 14),
            (9, 10),
            (9, 11),
            (10, 12),
            (11, 12),
            (11, 15),
            (12, 13),
            (12, 15),
            (12, 16),
        ]
    )
    G.add_edges_from(
        [
            (13, 14),
            (13, 16),
            (13, 17),
            (14, 17),
            (15, 16),
            (16, 17),
        ]
    )
    return G


def get_circulant_20_2():
    """Circulant graph with 20 vertices, each attached to 2 neighbors on either side."""
    return nx.circulant_graph(20, [1, 2])


def get_circulant_20_4():
    """Circulant graph with 20 vertices, each attached to 4 neighbors on either side."""
    return nx.circulant_graph(20, [1, 2, 3, 4])


def get_scale_free_100(seed=None):
    """Scale-free network with 100 vertices (Barabási-Albert, m=2)."""
    return nx.barabasi_albert_graph(100, 2, seed=seed)


def get_scale_free_410(seed=None):
    """Scale-free network with 410 vertices (Barabási-Albert, m=2)."""
    return nx.barabasi_albert_graph(410, 2, seed=seed)


def get_small_world_100(seed=None):
    """Small-world network with 100 vertices (Watts-Strogatz, k=5, p=0.3)."""
    return nx.watts_strogatz_graph(100, 5, 0.3, seed=seed)


def get_small_world_410(seed=None):
    """Small-world network with 410 vertices (Watts-Strogatz, k=3, p=0.3)."""
    return nx.watts_strogatz_graph(410, 3, 0.3, seed=seed)


def get_karate_graph():
    """Return the Zachary karate club network (34 nodes)."""
    path = "../data/karate.mtx"
    G = load_mtx_graph(path)
    return nx.convert_node_labels_to_integers(G)


def get_infect_dublin(path):
    """Load the infect-dublin network from an MTX file."""
    return load_mtx_graph(path)


def load_mtx_graph(path):
    """Load a graph from a Matrix Market (.mtx) file."""

    mat = scipy.io.mmread(path)
    mat = mat.tocoo()
    G = nx.Graph()
    for u, v in zip(mat.row, mat.col):
        if u != v:
            G.add_edge(int(u), int(v))
    # Keep largest connected component
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    return G

def select_clustered_seeds(
    G,
    n_seeds,
    seed_node=None,
    rng=None,
):
    """Select early adopters that are neighbors of each other (clustered seeding).
    Picks a seed node and expands to include its neighbors until we have n_seeds.
    """
    rng = rng or random.Random()
    nodes = list(G.nodes())
    if not nodes:
        return set()

    if seed_node is None:
        seed_node = rng.choice(nodes)
    elif seed_node not in G:
        raise ValueError(f"seed_node {seed_node} not in graph")

    seeds = {seed_node}
    candidates = list(G.neighbors(seed_node))

    while len(seeds) < n_seeds and candidates:
        # Prefer nodes that are neighbors of current seeds
        rng.shuffle(candidates)
        added = False
        for c in candidates:
            if c not in seeds:
                seeds.add(c)
                for n in G.neighbors(c):
                    if n not in seeds and n not in candidates:
                        candidates.append(n)
                candidates = [x for x in candidates if x not in seeds]
                added = True
                break
        if not added:
            break

    return seeds


def select_incubator_seeds(G, n_seeds, neighborhood_size=5, rng=None):
    """Select early adopters within an incubator neighborhood.
    Picks a random node and n_seeds-1 of its neighbors (or all if fewer).
    """
    rng = rng or random.Random()
    nodes = list(G.nodes())
    if not nodes:
        return set()

    center = rng.choice(nodes)
    neighbors = list(G.neighbors(center))
    rng.shuffle(neighbors)

    seeds = {center}
    take = min(n_seeds - 1, len(neighbors), neighborhood_size)
    for i in range(take):
        seeds.add(neighbors[i])

    return seeds


def select_random_seeds(
    G,
    n_seeds,
    rng=None,
):
    """Select early adopters uniformly at random."""
    rng = rng or random.Random()
    nodes = list(G.nodes())
    if n_seeds >= len(nodes):
        return set(nodes)
    return set(rng.sample(nodes, n_seeds))


class ComplexContagionPopulation:
    """
    Agent-based complex contagion model on a network.

    Agents have state B (Baseline) or A (Adopted). Early adopters start in A.
    An agent in B adopts (switches to A) when enough neighbors have adopted.
    Threshold can be absolute (at least k neighbors) or fractional (at least f fraction).
    """

    def __init__(
        self,
        G,
        early_adopters=None,
        threshold_type="absolute",
        threshold=2.0,
        seed=42,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.G = G
        self.threshold_type = threshold_type
        self.threshold = threshold

        # Initialize agent states
        self._state = {}
        for node in G.nodes():
            if early_adopters and node in early_adopters:
                self._state[node] = BehaviorState.ADOPTED
            else:
                self._state[node] = BehaviorState.BASELINE

    def get_state(self, node):
        """Return the current state of a node."""
        return self._state[node]

    def get_state_counts(self):
        """Return counts per state."""
        counts = {BehaviorState.BASELINE: 0, BehaviorState.ADOPTED: 0}
        for s in self._state.values():
            counts[s] += 1
        return counts

    def get_color_map(self):
        """Return node colors for plotting: B=blue, A=cyan (aquamarine)."""
        color_map = {
            BehaviorState.BASELINE: "blue",
            BehaviorState.ADOPTED: "cyan",
        }
        return [color_map[self._state[n]] for n in self.G.nodes()]

    def _should_adopt(self, node):
        """Check if a baseline agent should adopt based on neighbor states."""
        if self._state[node] == BehaviorState.ADOPTED:
            return False

        neighbors = list(self.G.neighbors(node))
        if not neighbors:
            return False

        n_adopted = sum(1 for n in neighbors if self._state[n] == BehaviorState.ADOPTED)

        if self.threshold_type == "absolute":
            return n_adopted >= int(self.threshold)
        else:  # fractional
            return (n_adopted / len(neighbors)) >= self.threshold

    def step(self):
        """
        Advance simulation by one time step.
        All baseline agents check their neighbors; those meeting threshold adopt.
        Returns counts per state after the step.
        """
        to_adopt = []
        for node in self.G.nodes():
            if self._should_adopt(node):
                to_adopt.append(node)

        for node in to_adopt:
            self._state[node] = BehaviorState.ADOPTED

        return self.get_state_counts()

    def run(
        self,
        max_steps=1000,
        stop_when_stable=True,
    ):
        """
        Run simulation for up to max_steps.
        Returns list of state counts per step.
        """
        history = []
        prev_counts = None

        for _ in range(max_steps):
            counts = self.step()
            history.append(counts.copy())

            if stop_when_stable and prev_counts is not None:
                if counts == prev_counts:
                    break
            prev_counts = counts

        return history

    def get_adoption_curve(self, history):
        """Extract number of adopted agents over time from history."""
        return [h[BehaviorState.ADOPTED] for h in history]


def network_summary(G):
    """Compute summary statistics for a network."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]

    return {
        "n_nodes": n,
        "n_edges": m,
        "density": nx.density(G),
        "avg_degree": sum(degrees) / n if n else 0,
        "clustering": nx.average_clustering(G),
        "avg_path_length": nx.average_shortest_path_length(G) if nx.is_connected(G) else float("nan"),
        "diameter": nx.diameter(G) if nx.is_connected(G) else -1,
    }
