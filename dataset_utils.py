import networkx as nx

G = nx.DiGraph()
with open("./dataset/OriginalLevels/WADs/Doom/Processed/E1M1.txt") as levelfile:
    for y, line in enumerate(levelfile):
        for x, tile in enumerate(line.strip()):
            G.add_node((x,y), {"tile":tile})
            if x > 0:
                G.add_edge((x,y),(x-1,y), {'direction':'W'})
                G.add_edge((x-1,y),(x,y), {'direction':'E'})
            if y > 0:
                G.add_edge((x,y),(x,y-1), {'direction':'N'})
                G.add_edge((x,y-1),(x,y), {'direction':'S'})


    print(G.number_of_nodes())
    print(G.number_of_edges())
    pass