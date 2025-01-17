import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_collatz_sequence(n, G=None, node_sizes=None, longest_sequence=None, longest_mono_sequence=None):
    if G is None:
        G = nx.DiGraph()
    if node_sizes is None:
        node_sizes = {}
    if longest_sequence is None:
        longest_sequence = {'length': 0, 'start': None}
    if longest_mono_sequence is None:
        longest_mono_sequence = {'length': 0, 'start': None, 'color': None}
    
    prev_n = n
    # Add initial number
    if not G.has_node(n):
        G.add_node(n)
        node_sizes[n] = 100
    else:
        node_sizes[n] = node_sizes.get(n, 100) + 100
    
    sequence_length = 1
    mono_sequence_length = 1
    mono_sequence_start = n
    mono_sequence_color = 'lightblue' if n % 2 == 0 else 'lightgreen'
    
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
        
        sequence_length += 1
        if n % 2 == 0 and mono_sequence_color == 'lightgreen':
            mono_sequence_length = 1
            mono_sequence_start = n
            mono_sequence_color = 'lightblue'
        elif n % 2 != 0 and mono_sequence_color == 'lightblue':
            mono_sequence_length = 1
            mono_sequence_start = n
            mono_sequence_color = 'lightgreen'
        else:
            mono_sequence_length += 1
        
        if sequence_length > longest_sequence['length']:
            longest_sequence['length'] = sequence_length
            longest_sequence['start'] = prev_n
        if mono_sequence_length > longest_mono_sequence['length']:
            longest_mono_sequence['length'] = mono_sequence_length
            longest_mono_sequence['start'] = mono_sequence_start
            longest_mono_sequence['color'] = mono_sequence_color
    
    return G, node_sizes, longest_sequence, longest_mono_sequence

def draw_collatz_tree(start, end):
    fig, ax = plt.subplots(figsize=(12, 10))
    G = nx.DiGraph()
    node_sizes = {}
    pos = None
    longest_sequence = {'length': 0, 'start': None}
    longest_mono_sequence = {'length': 0, 'start': None, 'color': None}
    
    def update(frame):
        nonlocal G, node_sizes, pos, longest_sequence, longest_mono_sequence
        ax.clear()
        
        if frame > 0:
            num = start + frame - 1
            G, node_sizes, longest_sequence, longest_mono_sequence = generate_collatz_sequence(num, G, node_sizes, longest_sequence, longest_mono_sequence)
            
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
        ax.set_title(f'Collatz Sequences: 1 to {start + max(0, frame-1)}\n'
                     f'Longest Sequence: {longest_sequence["length"]} from {longest_sequence["start"]}\n'
                     f'Longest Monochromatic Sequence: {longest_mono_sequence["length"]} from {longest_mono_sequence["start"]}, Color: {longest_mono_sequence["color"]}')
    
    frames = end - start + 2
    ani = animation.FuncAnimation(fig, update, frames=frames, 
                                interval=1000, repeat=False)
    plt.show()

# Visualize sequences from 1 to 100
draw_collatz_tree(1, 100)
