import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_collatz_sequence(n, G=None, node_sizes=None):
    if G is None:
        G = nx.DiGraph()
    if node_sizes is None:
        node_sizes = {}
    
    prev_n = n
    # Add initial number
    if not G.has_node(n):
        G.add_node(n)
        node_sizes[n] = 100
    else:
        node_sizes[n] = node_sizes.get(n, 100) + 100

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
            node_sizes[n] = node_sizes.get(n, 100) + 100
            
        G.add_edge(old_n, n)
    
    return G, node_sizes

def draw_collatz_tree(start, end):
    fig, ax = plt.subplots(figsize=(12, 10))
    G = nx.DiGraph()
    node_sizes = {}
    pos = None
    
    def update(frame):
        nonlocal G, node_sizes, pos
        ax.clear()
        
        if frame > 0:
            num = start + frame - 1
            G, node_sizes = generate_collatz_sequence(num, G, node_sizes)
            
        if len(G.nodes()) > 0:
            # Adjust repulsion and scale dynamically
            k = 2.5 / (len(G.nodes()) ** 0.5)  # Stronger repulsion for dense graphs
            scale = 50 + len(G.nodes()) * 2   # Dynamic scaling for larger graphs
            
            pos = nx.spring_layout(G, k=k, scale=scale, iterations=200, seed=42)
            
            node_colors = [('lightblue' if n % 2 == 0 else 'lightgreen') for n in G.nodes()]
            node_sizes_list = [node_sizes.get(n, 100) for n in G.nodes()]
            
            # Draw edges with proper margins
            for (src, dst) in G.edges():
                src_size = node_sizes.get(src, 100)
                dst_size = node_sizes.get(dst, 100)
                src_margin = (src_size ** 0.5) / 2
                dst_margin = (dst_size ** 0.5) / 2
                
                nx.draw_networkx_edges(G, pos, ax=ax,
                                     edgelist=[(src, dst)],
                                     width=2,
                                     edge_color='gray',
                                     arrows=True,
                                     min_source_margin=src_margin,
                                     min_target_margin=dst_margin)
            
            nx.draw_networkx_nodes(G, pos, ax=ax,
                                 node_size=node_sizes_list,
                                 node_color=node_colors,
                                 edgecolors='black',
                                 node_shape="o",
                                 alpha=0.8)
            
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        
        plt.axis('off')
        ax.set_title(f'Collatz Sequences: 1 to {start + max(0, frame-1)}')
    
    frames = end - start + 2
    ani = animation.FuncAnimation(fig, update, frames=frames, 
                                interval=1000, repeat=False)
    plt.show()

# Visualize sequences from 1 to 100
draw_collatz_tree(1, 100)
