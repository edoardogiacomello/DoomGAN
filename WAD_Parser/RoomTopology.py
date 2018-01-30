from skimage.future import graph as skg
from skimage import filters, color
from skimage.io import imshow
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.measure import label
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries
import networkx as nx
import numpy as np
from doomutils import vertices_to_segment_list

def _plot_room(rooms, node):
    from skimage.draw import line
    plt.figure()
    img = np.zeros_like(rooms)
    for l in node['walls']:
        seg = l[0]
        if l[1] == 0:
            color=255
        else:
            color = 127
        rr,cc = line(int(np.floor(seg[0][0])),int(np.floor(seg[0][1])),int(np.floor(seg[1][0])),int(np.floor(seg[1][1])))
        img[rr,cc] = color
    imshow(img)

def _plot_boundary_map(rooms, graph):
    from skimage.draw import line
    plt.figure()
    img = np.zeros_like(rooms)
    for n in graph.node:
        if n == 0:
            continue
        node = graph.node[n]
        for l in node['walls']:
            seg = l[0]
            if l[1] == 0:
                color=127
            elif l[1] is None:
                color = 255
            else:
                color = 75
            rr,cc = line(int(np.floor(seg[0][0])),int(np.floor(seg[0][1])),int(np.floor(seg[1][0])),int(np.floor(seg[1][1])))
            img[rr,cc] = color
    imshow(img)

def _reverse_wall(wall):
    """
    Return the same wall with swapped extrema:
    :param wall:  (((x1, y1), (x2, y2)), destination_node)
    :return: (((x2, y2), (x1, y1)), destination_node)
    """
    return tuple(reversed(wall[0])), wall[1]

def create_graph(floormap, return_dist=False, room_coordinates=False):
    """
    Segment the floormap into rooms and create a Region Adjacency Graph for the level.
    Many settings for decorating the graph are possible, by default the simplest form is returned.
    :param floormap: Binary image representing the level floor
    :param return_dist: If true, also returns the distance map of each point to the closest wall.
    :param room_coordinates: If true, each graph node will contain the room vertices and information about the walls
    :return: (Roommap, Graph) if return_dist is False, else (Roommap, Graph, dist)
    """
    # Ensuring that floormap is always a boolean array
    floormap = floormap.astype(np.bool)

    dist = ndi.distance_transform_edt(floormap)
    threshold = int(dist.max())
    optimal_threshold = 0
    number_of_centers = 0
    # Finding room center and finding the optimal threshold (the one that maximizes the number of rooms)
    for i in range(int(dist.max()), int(dist.min())-1,-1):
       local_max = peak_local_max(dist, threshold_abs=threshold-i, indices=False, labels=floormap, min_distance=3)
       markers = ndi.label(local_max)[0]
       if markers.max() > number_of_centers:
          optimal_threshold = threshold-i
          number_of_centers = markers.max()

    # Computing roommap with the optimal threshold
    local_max = peak_local_max(dist, min_distance=3, indices=False, labels=floormap, threshold_abs=optimal_threshold)
    markers = ndi.label(local_max)[0]
    roommap = watershed(-dist, markers, mask=floormap)

    room_RAG_boundaries = skg.rag_boundary(roommap, filters.sobel(color.rgb2gray(roommap)))

    if room_coordinates:
        # For each floor...
        floors = label(floormap)
        for floor_id in range(max(1, floors.min()), floors.max() + 1):  # Skipping label 0 (background)
            # Building the wall list for floor boundaries
            walls_vertices = [tuple(v) for v in
                              find_contours((floors == floor_id) * 1, 0.5, positive_orientation='low')[0]]
            floor_boundaries = tuple(vertices_to_segment_list(walls_vertices))
            # Map of rooms belonging to current floor
            rooms = roommap * (floors == floor_id)
            for room_id in range(max(1, rooms.min()), rooms.max() + 1):  # Skipping label 0 (background)
                if room_id not in rooms:
                    # Some room id may be in another floor, if they are enumerated horizontally
                    continue
                rooms_vertices = [tuple(v) for v in
                                  find_contours((rooms == room_id) * 1, 0.5, positive_orientation='low')[0]]
                room_boundaries = tuple(vertices_to_segment_list(rooms_vertices))


                room_RAG_boundaries.node[room_id]['walls'] = list()
                for segment in room_boundaries:
                    leads_to = 0 if segment in floor_boundaries else None # We cannot still know edges for other rooms but background
                    room_RAG_boundaries.node[room_id]['walls'].append((segment, leads_to))

            # Here we still miss the relation between boundary and edges.
            # Second pass
            for room_id in range(max(1, rooms.min()), rooms.max() + 1):
                if room_id not in rooms:
                    # Some room id may be in another floor, if they are enumerated horizontally
                    continue
                boundaries_current = {wall for wall in room_RAG_boundaries.node[room_id]['walls'] if wall[1] is None}
                for neigh in room_RAG_boundaries.adj[room_id]:
                    if neigh == 0:
                        continue
                    # Finding the neighbour boundaries. We must consider both directions for each vertex
                    boundaries_neigh = {wall for wall in room_RAG_boundaries.node[neigh]['walls'] if wall[1] is None}
                    boundaries_neigh_reverse = {_reverse_wall(wall) for wall in room_RAG_boundaries.node[neigh]['walls'] if wall[1] is None}

                    common_segments = boundaries_current.intersection(boundaries_neigh)
                    common_segments_reversed = boundaries_current.intersection(boundaries_neigh_reverse)
                    # Marking the boundary in the two nodes with the destination node
                    # Each node will contain the list
                    for cs in common_segments:
                        i_current = room_RAG_boundaries.node[room_id]['walls'].index(cs)
                        i_neighbour = room_RAG_boundaries.node[neigh]['walls'].index(cs)
                        room_RAG_boundaries.node[room_id]['walls'][i_current] = (cs[0], neigh)
                        room_RAG_boundaries.node[neigh]['walls'][i_neighbour] = (cs[0], room_id)
                    # Same thing in the case of reversed segments
                    for cs in common_segments_reversed:
                        rev_cs = _reverse_wall(cs)
                        i_current = room_RAG_boundaries.node[room_id]['walls'].index(cs)
                        i_neighbour = room_RAG_boundaries.node[neigh]['walls'].index(rev_cs)
                        room_RAG_boundaries.node[room_id]['walls'][i_current] = (cs[0], neigh)
                        room_RAG_boundaries.node[neigh]['walls'][i_neighbour] = (rev_cs[0], room_id)
        #_plot_boundary_map(rooms, room_RAG_boundaries)



    if return_dist:
        return roommap, room_RAG_boundaries, dist
    return roommap, room_RAG_boundaries

def topological_features(floormap, prepare_for_doom=False):
    roommap, room_graph, dist = create_graph(floormap, return_dist=True, room_coordinates=prepare_for_doom)
    room_props = regionprops(roommap)
    for r in range(1, roommap.max() + 1):
        # Room Size
        room_graph.node[r]["area"] = room_props[r - 1]["area"]
        room_graph.node[r]["perimeter"] = room_props[r - 1]["perimeter"]
        mask = (roommap == r)
        max_dist = np.max(mask * dist)
        centroid_shape = (dist * mask) == max_dist
        room_graph.node[r]["relative_eccentricity"] = np.count_nonzero(centroid_shape)
        room_graph.node[r]["relative_radius"] = max_dist
        room_graph.node[r]["form_factor"] = room_graph.node[r]["relative_eccentricity"] / (
            room_graph.node[r]["relative_radius"] ** 2)
        room_graph.node[r]["centroid"] = room_props[r - 1]["centroid"]

        if room_graph.node[r]["form_factor"] >= 1:
            room_graph.node[r]["type"] = "corridor"
        else:
            room_graph.node[r]["type"] = "room"
                # TODO: Merge as many info as possible into the graph, as distance, enemies, etc.


    centroid_distance = dict()
    for i, j in room_graph.edges():
        # Decorate the edges with the distance
        if i==0 or j == 0:
            continue
        centroid_distance[(i,j)] = np.linalg.norm(np.asarray(room_graph.node[i]["centroid"])-np.asarray(room_graph.node[j]["centroid"])).item()
    nx.set_edge_attributes(room_graph, 'centroid_distance', centroid_distance)
    return roommap, room_graph
