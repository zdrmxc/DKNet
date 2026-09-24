import numpy as np


def get_hop_distance(num_node: int, edge: list, max_hop: int = 1) -> np.ndarray:
    """
    Computes the shortest hop-distance matrix between all node pairs.

    Args:
        num_node (int): Total number of skeletal joints.
        edge (list): List of graph edge tuples (i, j).
        max_hop (int): Maximum hop distance to consider (default: 1).

    Returns:
        np.ndarray: Shortest hop distance matrix of shape [num_node, num_node].
    """
    adj = np.zeros((num_node, num_node))
    for i, j in edge:
        adj[j, i] = 1.0
        adj[i, j] = 1.0

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(adj, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0

    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d

    return hop_dis


def normalize_digraph(A: np.ndarray) -> np.ndarray:
    """
    Performs column degree normalization on an adjacency matrix: D^{-1} A.

    Args:
        A (np.ndarray): Unnormalized adjacency matrix of shape [N, N].

    Returns:
        np.ndarray: Normalized adjacency matrix.
    """
    dl = np.sum(A, axis=0)
    num_node = A.shape[0]
    dn = np.zeros((num_node, num_node))

    for i in range(num_node):
        if dl[i] > 0:
            dn[i, i] = dl[i] ** (-1)

    return np.dot(A, dn)


class Graph:
    """
    Single-frame spatial skeletal topology graph for 3D human pose estimation.
    Configured for the Human3.6M 17-joint skeletal layout.

    Partitions the spatial graph into K=4 distinct adjacency sub-matrices:
        - A[0]: Self-connection (A0, hop = 0)
        - A[1]: Centripetal / inward aggregation toward torso hub (A1, hop = 1)
        - A[2]: Centrifugal / outward propagation toward extremities (A2, hop = 1)
        - A[3]: Cross-body bilateral mirror symmetric connections (A3, hop = 1)

    Args:
        layout (str): Skeleton joint layout (default: 'hm36_gt').
        strategy (str): Graph partitioning strategy (default: 'spatial').
        pad (int): Temporal padding size (default: 0 for single-frame models).
        max_hop (int): Maximum graph hop distance (default: 1).
        dilation (int): Dilation rate for hop distance (default: 1).
    """
    def __init__(self, layout: str = 'hm36_gt', strategy: str = 'spatial',
                 pad: int = 0, max_hop: int = 1, dilation: int = 1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node = 17

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.dist_center = self.get_distance_to_center(layout)
        self.A = self.get_adjacency(strategy)

    def get_distance_to_center(self, layout: str) -> np.ndarray:
        """
        Defines the hierarchical topological distance from each joint to the pelvic root center.
        """
        dist_center = np.zeros(self.num_node)
        # Lower body (hips, knees, ankles)
        dist_center[0:7] = [1, 2, 3, 4, 2, 3, 4]
        # Spine and head
        dist_center[7:11] = [0, 1, 2, 3]
        # Upper limbs (shoulders, elbows, wrists)
        dist_center[11:17] = [2, 3, 4, 2, 3, 4]
        return dist_center

    def get_edge(self, layout: str):
        """
        Constructs physical kinematic bone connections and bilateral symmetric pairs.
        """
        # Standard physical kinematic skeletal connections
        neighbour_base = [
            (0, 1), (2, 1), (3, 2),
            (4, 0), (5, 4), (6, 5),
            (7, 0), (8, 7), (9, 8), (10, 9),
            (11, 8), (12, 11), (13, 12),
            (14, 8), (15, 14), (16, 15)
        ]

        # Bilateral mirror-symmetric limb joint pairs (left <-> right)
        sym_base = [
            (6, 3),    # Left ankle <-> Right ankle
            (5, 2),    # Left knee <-> Right knee
            (4, 1),    # Left hip <-> Right hip
            (11, 14),  # Left shoulder <-> Right shoulder
            (12, 15),  # Left elbow <-> Right elbow
            (13, 16)   # Left wrist <-> Right wrist
        ]

        self.self_link = [(i, i) for i in range(self.num_node)]
        self.neighbour_link = neighbour_base
        self.sym_link = sym_base
        self.edge = self.self_link + self.neighbour_link + self.sym_link

    def get_adjacency(self, strategy: str) -> np.ndarray:
        """
        Partitions and computes the spatial adjacency tensor [4, 17, 17].
        """
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1.0
        normalize_adjacency = normalize_digraph(adjacency)

        A = []
        for hop in valid_hop:
            a_root = np.zeros((self.num_node, self.num_node))
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            a_sym = np.zeros((self.num_node, self.num_node))

            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[j, i] == hop:
                        # Symmetric mirror links
                        if (j, i) in self.sym_link or (i, j) in self.sym_link:
                            a_sym[j, i] = normalize_adjacency[j, i]
                        # Root / self connections
                        elif self.dist_center[j] == self.dist_center[i]:
                            a_root[j, i] = normalize_adjacency[j, i]
                        # Inward / centripetal aggregation (closer to torso)
                        elif self.dist_center[j] > self.dist_center[i]:
                            a_close[j, i] = normalize_adjacency[j, i]
                        # Outward / centrifugal propagation (further from torso)
                        else:
                            a_further[j, i] = normalize_adjacency[j, i]

            if hop == 0:
                A.append(a_root)
            else:
                A.append(a_close)
                A.append(a_further)
                A.append(a_sym)

        return np.stack(A).astype(np.float32)