import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

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

def draw_collatz_mst_3d(start, end):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
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
            # 3D spring layout
            pos = nx.spring_layout(G, dim=3, k=2.5 / (len(G.nodes()) ** 0.5), iterations=200, seed=42)
            
            # Extract 3D coordinates
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_z = [pos[n][2] for n in G.nodes()]
            
            # Draw MST edges
            for (src, dst, _) in mst_edges:
                x = [pos[src][0], pos[dst][0]]
                y = [pos[src][1], pos[dst][1]]
                z = [pos[src][2], pos[dst][2]]
                ax.plot(x, y, z, color='green', linewidth=2, alpha=0.8)
            
            # Draw nodes
            node_colors = [('lightblue' if n % 2 == 0 else 'lightgreen') for n in G.nodes()]
            node_sizes_list = [node_sizes.get(n, 100) for n in G.nodes()]
            ax.scatter(node_x, node_y, node_z, s=node_sizes_list, c=node_colors, edgecolors='black', alpha=0.8)
            
            # Draw labels
            for n in G.nodes():
                ax.text(pos[n][0], pos[n][1], pos[n][2], str(n), fontsize=8, color='black')
        
        ax.set_title(f'3D Minimum Spanning Tree of Collatz Sequences: 1 to {start + max(0, frame-1)}')
        ax.set_axis_off()

    frames = end - start + 2
    ani = animation.FuncAnimation(fig, update, frames=frames, 
                                  interval=1000, repeat=False)
    plt.show()

# Visualize the MST for Collatz sequences from 1 to 100 in 3D
draw_collatz_mst_3d(1, 100)
