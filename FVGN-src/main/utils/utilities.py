import numpy as np
from torch_geometric.data import Data
import enum


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9
    REINFORCED_BOUNDARY = 10


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def decompose_and_trans_node_attr_to_cell_attr_graph(
    graph, has_changed_node_attr_to_cell_attr
):
    # graph: torch_geometric.data.data.Data
    # TODO: make it more robust
    x, edge_index, edge_attr, global_attr = None, None, None, None
    if has_changed_node_attr_to_cell_attr:
        for key in graph.keys:
            if key == "x":
                x = graph.x  # avoid exception
            elif key == "edge_index":
                edge_index = graph.edge_index
            elif key == "edge_attr":
                edge_attr = graph.edge_attr
            elif key == "global_attr":
                global_attr = graph.global_attr
            elif key == "cell_face":
                cell_face = graph.cell_face
            else:
                pass
        return (x, edge_index, edge_attr, global_attr)
    else:
        for key in graph.keys:
            if key == "x":
                x = graph.x  # avoid exception
            elif key == "edge_index":
                edge_index = graph.edge_index
            elif key == "edge_attr":
                edge_attr = graph.edge_attr
            elif key == "global_attr":
                global_attr = graph.global_attr
            elif key == "cell_face":
                cell_face = graph.cell_face
            else:
                pass
        return (x, edge_index, edge_attr, global_attr)


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def copy_geometric_data(graph, has_changed_node_attr_to_cell_attr):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    cell_attr, edge_index, edge_attr, global_attr = (
        decompose_and_trans_node_attr_to_cell_attr_graph(
            graph, has_changed_node_attr_to_cell_attr
        )
    )

    ret = Data(x=cell_attr, edge_index=edge_index, edge_attr=edge_attr)
    ret.global_attr = global_attr

    return ret


def shuffle_np(array):
    array_t = array.copy()
    np.random.shuffle(array_t)
    return array_t
