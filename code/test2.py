import osmnx as ox
G = ox.graph_from_place('Hoan Kiem, Ha Noi, Viet Nam', network_type='drive')
ox.plot_graph(G)