# Modified DAVIS interactive robot https://github.com/albertomontesg/davis-interactive
# I tamed their robot to work as a slave for me

"""
TamedRobot: Automated Point Sampling from Error Regions for Scribble Generation

This module implements an automated scribble generation algorithm that samples points
from error regions (false positives/negatives in segmentation masks) to create 
natural-looking scribbles for interactive segmentation training.

Point Sampling Strategy:
1. Skeleton Generation: Extract the medial axis (skeleton) of the error region
2. Graph Construction: Convert skeleton pixels into a graph with connected components
3. Path Finding: Find the longest paths in each connected component
4. Bezier Curve Sampling: Sample points along bezier curves fitted to the paths
5. Line Rendering: Use Bresenham's algorithm to create continuous pixel lines

The algorithm creates smooth, realistic scribbles that follow the shape of error regions,
mimicking human annotation behavior.
"""

import time
import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.special import comb
from skimage.filters import rank
from skimage.morphology import dilation, disk, erosion, medial_axis
from sklearn.neighbors import radius_neighbors_graph
import cv2


def bezier_curve(points, nb_points=1000):
    """ Given a list of points compute a bezier curve from it.
    
    This function performs point sampling by creating a smooth bezier curve through
    the given control points. The sampling is uniform along the parametric curve
    using parameter values t ∈ [0, 1].
    
    Point Sampling Method:
    - Uses Bernstein polynomials to interpolate between control points
    - Samples nb_points uniformly spaced points along the parametric curve
    - Each point is computed as: P(t) = Σ B(n,i,t) * P_i
      where B(n,i,t) = C(n,i) * t^(n-i) * (1-t)^i is the Bernstein basis
    
    # Arguments
        points: ndarray. Array of points with shape (N, 2) with N being the
            number of points and the second dimension representing the
            (x, y) coordinates.
        nb_points: Integer. Number of points to sample from the bezier curve.
            This value must be larger than the number of points given in
            `points`. Maximum value 10000. Default is 1000, which provides
            smooth curves for most scribble generation tasks.

    # Returns
        ndarray: Array of shape (nb_points, 2) with the sampled points along
            the bezier curve.

    """
    nb_points = min(nb_points, 1000)

    points = np.asarray(points, dtype=np.float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            '`points` should be two dimensional and have shape: (N, 2)')

    n_points = len(points)
    if n_points > nb_points:
        # We are downsampling points
        return points

    t = np.linspace(0., 1., nb_points).reshape(1, -1)

    # Compute the Bernstein polynomial of n, i as a function of t
    i = np.arange(n_points).reshape(-1, 1)
    n = n_points - 1
    polynomial_array = comb(n, i) * (t**(n - i)) * (1 - t)**i

    bezier_curve_points = polynomial_array.T.dot(points)

    return bezier_curve_points


def bresenham(points):
    """ Apply Bresenham algorithm for a list points.
    
    This algorithm creates continuous pixel lines between consecutive points,
    ensuring no gaps in the final scribble. It's a critical step in converting
    the sampled bezier curve points into a drawable mask.
    
    Point Sampling Behavior:
    - For each consecutive pair of points, interpolates all pixels along the line
    - Uses integer arithmetic for efficiency
    - Guarantees 8-connected pixel connectivity (no diagonal gaps)
    - The resulting points form a continuous, rasterized line suitable for
      creating binary scribble masks

    More info: https://en.wikipedia.org/wiki/Bresenham's_line_algorithm

    # Arguments
        points: ndarray. Array of points with shape (N, 2) with N being the number
            if points and the second coordinate representing the (x, y)
            coordinates.

    # Returns
        ndarray: Array of points after having applied the bresenham algorithm.
            Contains all pixels along the lines connecting consecutive input points.
    """

    points = np.asarray(points, dtype=np.int)

    def line(x0, y0, x1, y1):
        """ Bresenham line algorithm.
        """
        d_x = x1 - x0
        d_y = y1 - y0

        x_sign = 1 if d_x > 0 else -1
        y_sign = 1 if d_y > 0 else -1

        d_x = np.abs(d_x)
        d_y = np.abs(d_y)

        if d_x > d_y:
            xx, xy, yx, yy = x_sign, 0, 0, y_sign
        else:
            d_x, d_y = d_y, d_x
            xx, xy, yx, yy = 0, y_sign, x_sign, 0

        D = 2 * d_y - d_x
        y = 0

        line = np.empty((d_x + 1, 2), dtype=points.dtype)
        for x in range(d_x + 1):
            line[x] = [x0 + x * xx + y * yx, y0 + x * xy + y * yy]
            if D >= 0:
                y += 1
                D -= 2 * d_x
            D += 2 * d_y

        return line

    nb_points = len(points)
    if nb_points < 2:
        return points

    new_points = []

    for i in range(nb_points - 1):
        p = points[i:i + 2].ravel().tolist()
        new_points.append(line(*p))

    new_points = np.concatenate(new_points, axis=0)

    return new_points


class TamedRobot(object):
    """
    Automated scribble generator that samples points from error regions.
    
    This class implements a sophisticated point sampling algorithm for generating
    realistic scribbles from segmentation error regions. The sampling process
    follows these steps:
    
    1. Skeleton Extraction (_generate_scribble_mask):
       - Applies morphological operations to smooth the error region
       - Extracts the medial axis (skeleton) - a thin representation of the shape
       - The skeleton provides initial candidate points for sampling
    
    2. Graph Construction (_mask2graph):
       - Converts skeleton pixels into graph nodes
       - Connects nearby pixels (within sqrt(2) distance) with weighted edges
       - Edge weights represent distances between pixels
    
    3. Path Selection (_acyclics_subgraphs, _longest_path_in_tree):
       - Decomposes the graph into connected components
       - Removes cycles by pruning high-weight edges
       - Finds the longest path in each tree-like component
       - Longer paths produce more visible, meaningful scribbles
    
    4. Bezier Curve Sampling (bezier_curve):
       - Fits smooth bezier curves through the selected path points
       - Samples nb_points (default 1000) uniformly along each curve
       - Creates natural-looking, smooth scribbles
    
    5. Line Rendering (bresenham):
       - Converts sampled points into continuous pixel lines
       - Ensures no gaps in the final scribble mask
    
    Parameters:
        kernel_size: float. Proportion of the region's side length used for
                     morphological operations (default 0.15)
        max_kernel_radius: int. Maximum radius for morphological kernels
                          (default 16 pixels)
        min_nb_nodes: int. Minimum nodes required in a graph component
                     (default 4)
        nb_points: int. Number of points to sample along each bezier curve
                  (default 1000). Higher values create smoother curves but
                  increase computation time.
    """

    def __init__(self,
                 kernel_size=.15,
                 max_kernel_radius=16,
                 min_nb_nodes=4,
                 nb_points=1000):

        self.kernel_size = kernel_size
        self.max_kernel_radius = max_kernel_radius
        self.min_nb_nodes = min_nb_nodes
        self.nb_points = nb_points

    def _generate_scribble_mask(self, mask):
        """ Generate the skeleton from a mask
        
        Step 1 of Point Sampling: Extract candidate points from the error region.
        
        This method performs morphological operations to smooth and thin the error
        region into a skeleton (medial axis), which represents the "centerline" of
        the shape. The skeleton provides initial point samples that capture the
        region's topology.
        
        Algorithm:
        1. Calculate adaptive kernel size based on region area (kernel_radius ∝ sqrt(area))
        2. Apply morphological minimum then maximum (close operation) to remove noise
        3. Compute medial axis - the set of points equidistant from boundaries
        4. The resulting skeleton pixels become candidate points for scribble generation
        
        Given an error mask, the medial axis is computed to obtain the
        skeleton of the objects. In order to obtain smoother skeleton and
        remove small objects, an erosion and dilation operations are performed.
        The kernel size used is proportional the squared of the area.

        # Arguments
            mask: Numpy Array. Error mask (binary, with 1s indicating errors)

        Returns:
            skel: Numpy Array. Skeleton mask (binary, with 1s indicating skeleton pixels)
                  These pixels serve as initial point samples for the scribble.
        """
        mask = np.asarray(mask, dtype=np.uint8)
        side = np.sqrt(np.sum(mask > 0))

        mask_ = mask
        # kernel_size = int(self.kernel_size * side)
        kernel_radius = self.kernel_size * side * .5
        kernel_radius = min(kernel_radius, self.max_kernel_radius)

        compute = True
        while kernel_radius > 1. and compute:
            kernel = disk(kernel_radius)
            mask_ = rank.minimum(mask.copy(), kernel)
            mask_ = rank.maximum(mask_, kernel)
            compute = False
            if mask_.astype(np.bool).sum() == 0:
                compute = True
                prev_kernel_radius = kernel_radius
                kernel_radius *= .9

        mask_ = np.pad(
            mask_, ((1, 1), (1, 1)), mode='constant', constant_values=False)
        skel = medial_axis(mask_.astype(np.bool))
        skel = skel[1:-1, 1:-1]
        return skel

    def _mask2graph(self, skeleton_mask):
        """ Transforms a skeleton mask into a graph
        
        Step 2 of Point Sampling: Organize skeleton points into a connected graph.
        
        This method converts the discrete skeleton pixels into a graph structure,
        where each skeleton pixel becomes a node and nearby pixels are connected
        with edges. This enables path-finding algorithms to extract meaningful
        scribble trajectories.
        
        Algorithm:
        1. Extract (x,y) coordinates of all skeleton pixels
        2. Build a radius neighbors graph - connect pixels within sqrt(2) distance
           (sqrt(2) allows 8-connectivity: horizontal, vertical, and diagonal neighbors)
        3. Edge weights represent Euclidean distances between connected pixels
        4. Convert to NetworkX graph for path analysis

        Args:
            skeleton_mask (ndarray): Skeleton mask (binary)

        Returns:
            tuple(nx.Graph, ndarray): Returns a tuple where the first element
                is a Graph with skeleton pixels as nodes and the second element 
                is an array of xy coordinates indicating the coordinates for each 
                Graph node.

                If an empty mask is given, None is returned.
        """
        mask = np.asarray(skeleton_mask, dtype=np.bool)
        if np.sum(mask) == 0:
            return None

        h, w = mask.shape
        x, y = np.arange(w), np.arange(h)
        X, Y = np.meshgrid(x, y)

        X, Y = X.ravel(), Y.ravel()
        M = mask.ravel()

        X, Y = X[M], Y[M]
        points = np.c_[X, Y]
        G = radius_neighbors_graph(points, np.sqrt(2), mode='distance')
        T = nx.from_scipy_sparse_matrix(G)

        return T, points

    def _acyclics_subgraphs(self, G):
        """ Divide a graph into connected components subgraphs
        
        Step 3a of Point Sampling: Decompose the graph and remove cycles.
        
        Cycles in the skeleton graph represent regions where multiple paths exist.
        For clean scribble generation, we want tree-like structures (acyclic graphs)
        where there's a single path between any two points. This step breaks cycles
        by removing the highest-weight (longest) edge in each cycle.
        
        Algorithm:
        1. Decompose graph into connected components
        2. For each component, find cycles using NetworkX
        3. Remove the edge with maximum weight (distance) in each cycle
        4. Repeat until the graph is acyclic (tree-like)
        5. The resulting trees enable unambiguous longest-path extraction
        
        Divide a graph into connected components subgraphs and remove its
        cycles removing the edge with higher weight inside the cycle. Also
        prune the graphs by number of nodes in case the graph has not enought
        nodes.

        Args:
            G (nx.Graph): Graph

        Returns:
            list(nx.Graph): Returns a list of graphs which are subgraphs of G
                with cycles removed. Each subgraph is a tree suitable for
                path extraction.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        S = []  # List of subgraphs of G

        for c in nx.connected_components(G):
            g = G.subgraph(c).copy()

            # Remove all cycles that we may find
            has_cycles = True
            while has_cycles:
                try:
                    cycle = nx.find_cycle(g)
                    weights = np.asarray([G[u][v]['weight'] for u, v in cycle])
                    idx = weights.argmax()
                    # Remove the edge with highest weight at cycle
                    g.remove_edge(*cycle[idx])
                except nx.NetworkXNoCycle:
                    has_cycles = False

            S.append(g)

        return S

    def _longest_path_in_tree(self, G):
        """ Given a tree graph, compute the longest path and return it
        
        Step 3b of Point Sampling: Find the longest path in each tree component.
        
        The longest path represents the most significant feature of the error region,
        spanning from one end to the other. This creates the most informative scribble
        that best captures the region's extent and shape.
        
        Algorithm (Two-pass approach):
        1. Start from an arbitrary node v
        2. Find the furthest node v' from v using shortest path (in a tree,
           shortest path = only path = longest path in terms of number of edges)
        3. From v', find the longest path to any other node
        4. This path is guaranteed to be the diameter (longest path) of the tree
        
        The points along this path become control points for bezier curve fitting.
        
        Given an undirected tree graph, compute the longest path and return it.

        The approach use two shortest path transversals (shortest path in a
        tree is the same as longest path). This could be improve but would
        require implement it:
        https://cs.stackexchange.com/questions/11263/longest-path-in-an-undirected-tree-with-only-one-traversal

        Args:
            G (nx.Graph): Graph which should be an undirected tree graph

        Returns:
            list(int): Returns a list of indexes of the nodes belonging to the
                longest path. These node indices correspond to (x,y) coordinates
                that will be used as control points for bezier curve sampling.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        if not nx.is_tree(G):
            raise ValueError('Graph G must be a tree (graph without cycles)')

        # Compute the furthest node to the random node v
        v = list(G.nodes())[0]
        distance = nx.single_source_shortest_path_length(G, v)
        vp = max(distance.items(), key=lambda x: x[1])[0]
        # From this furthest point v' find again the longest path from it
        distance = nx.single_source_shortest_path(G, vp)
        longest_path = max(distance.values(), key=len)
        # Return the longest path

        return list(longest_path)

    def interact(self, error_mask):
        """
        Main method: Generate scribbles from an error mask through point sampling.
        
        This is the complete point sampling pipeline that converts an error region
        into a scribble mask. The algorithm follows these steps:
        
        1. Downsample the error mask (2x) for faster processing
        2. Generate skeleton using medial axis (extract initial point samples)
        3. Convert skeleton to graph (organize points into connected structure)
        4. Decompose into acyclic subgraphs (prepare for path finding)
        5. Find longest paths in each component (select most significant points)
        6. Fit bezier curves through path points (smooth interpolation)
        7. Sample points along bezier curves (nb_points=1000 per curve)
        8. Apply Bresenham algorithm (create continuous pixel lines)
        9. Upsample back to original resolution (2x)
        
        The result is a binary mask containing smooth, continuous scribbles that
        follow the shape of the error region, suitable for training interactive
        segmentation models.
        
        Args:
            error_mask: Binary mask indicating segmentation errors (false positives
                       or false negatives). Shape: (H, W)
        
        Returns:
            out_mask: Binary scribble mask with the same shape as input.
                     Contains 1s along the sampled scribble lines, 0s elsewhere.
        """
        # start_time = time.time()
        out_mask = np.zeros_like(error_mask)

        # Downsample for faster processing
        error_mask = (cv2.resize(error_mask, dsize=None, fx=(0.5), fy=(0.5), interpolation=cv2.INTER_AREA)>0.5).astype(np.uint8)

        # Generate scribbles
        skel_mask = self._generate_scribble_mask(error_mask)
        # skel_time = time.time() - start_time
        # print('skel', skel_time)

        graph_out = self._mask2graph(skel_mask)
        if graph_out is None:
            return out_mask
        G, P = graph_out
        # mask2graph_time = time.time() - start_time - skel_time
        # print('m2g time', mask2graph_time)

        # t_start = time.time()
        S = self._acyclics_subgraphs(G)
        # t = (time.time() - t_start)
        # print('asub time', t)

        # t_start = time.time()
        longest_paths_idx = [self._longest_path_in_tree(s) for s in S]
        longest_paths = [P[idx] for idx in longest_paths_idx]
        # t = (time.time() - t_start) 
        # print('longest time', t)

        # t_start = time.time()
        # Step 4: Fit bezier curves and sample points
        # For each longest path, create a smooth bezier curve and sample nb_points
        # along it. This converts the discrete skeleton path into a dense, smooth
        # set of points that will form a natural-looking scribble.
        scribbles_paths = [
            bezier_curve(p, self.nb_points) for p in longest_paths
        ]
        # t = (time.time() - t_start) 
        # print('asub time', t)

        # t_start = time.time()
        # Step 5: Convert sampled points to pixel masks using Bresenham
        # The bezier-sampled points are at arbitrary floating-point coordinates.
        # Bresenham's algorithm converts them into discrete, connected pixel
        # coordinates. We also upsample by 2x here to restore original resolution.
        for path in scribbles_paths:
            # Re-upsample the line (was downsampled 2x earlier)
            path = bresenham(path*2)
            out_mask[path[:, 1], path[:, 0]] = 1
        # t = (time.time() - t_start) 
        # print('final time', t)

        return out_mask

