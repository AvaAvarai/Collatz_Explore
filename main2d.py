import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_collatz_sequence(n, G=None, node_sizes=None):
    if G is None:
        G = nx.DiGraph()
    if node_sizes is None:
        node_sizes = {}
    
    # Add initial node
    if not G.has_node(n):
        G.add_node(n)
        node_sizes[n] = 100
    else:
        node_sizes[n] += 100

    while n != 1:
        old_n = n
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1

        if not G.has_node(n):
            G.add_node(n)
            node_sizes[n] = 100
        else:
            node_sizes[n] += 100

        # Assign a weight of 1 to edges
        G.add_edge(old_n, n, weight=1)
    
    return G, node_sizes

def draw_collatz_mst(start, end):
    fig, ax = plt.subplots(figsize=(12, 10))
    G = nx.DiGraph()
    node_sizes = {}
    pos = None
    mst_edges = []
    
    def update(frame):
        nonlocal G, node_sizes, pos, mst_edges
        ax.clear()
        
        if frame > 0:
            num = start + frame - 1
            G, node_sizes = generate_collatz_sequence(num, G, node_sizes)
            
            # Generate MST from the undirected graph
            undirected_G = G.to_undirected()
            mst = nx.minimum_spanning_tree(undirected_G)
            mst_edges = mst.edges(data=True)
        
        if len(G.nodes()) > 0:
            k = 2.5 / (len(G.nodes()) ** 0.5)
            scale = 50 + len(G.nodes()) * 2
            pos = nx.spring_layout(G, k=k, scale=scale, iterations=200, seed=42)
            
            # Draw MST edges
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(mst_edges),
                                   width=2, edge_color='green', alpha=0.8)
            
            # Draw nodes
            node_colors = [('lightblue' if n % 2 == 0 else 'lightgreen') for n in G.nodes()]
            node_sizes_list = [node_sizes.get(n, 100) for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, ax=ax,
                                   node_size=node_sizes_list,
                                   node_color=node_colors,
                                   edgecolors='black',
                                   alpha=0.8)
            
            # Draw node labels
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        
        plt.axis('off')
        ax.set_title(f'Minimum Spanning Tree of Collatz Sequences: 1 to {start + max(0, frame-1)}')
    
    frames = end - start + 2
    ani = animation.FuncAnimation(fig, update, frames=frames, 
                                  interval=1000, repeat=False)
    plt.show()

# Visualize the MST for Collatz sequences from 1 to 100
draw_collatz_mst(1, 100)
