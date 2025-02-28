# This experiment visualizes the spectral evolution of Collatz graphs by analyzing how the eigenvalues 
# of the graph Laplacian change as more numbers are added to the graph. For each number n from 1 to n_max, 
# it constructs an undirected graph where nodes represent numbers in Collatz sequences and edges connect 
# consecutive numbers in these sequences. The program computes the spectrum (eigenvalues) of each graph's 
# Laplacian matrix and creates an animation showing how these eigenvalues evolve, potentially revealing 
# structural patterns in the Collatz conjecture. The computation is parallelized for efficiency and the 
# result is saved as an MP4 video file.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os

def collatz_sequence(n):
    """Generate the Collatz sequence starting from n and collapse nodes."""
    sequence = set()
    while n != 1:
        sequence.add(n)
        n = n // 2 if n % 2 == 0 else 3 * n + 1
    sequence.add(1)
    return sequence

def build_collatz_graph(n_max):
    """Constructs the undirected, collapsed Collatz graph for numbers up to n_max."""
    G = nx.Graph()
    G.add_node(1)  # Prevent empty graph errors
    
    for n in range(1, n_max + 1):
        sequence = list(collatz_sequence(n))
        for i in range(len(sequence) - 1):
            G.add_edge(sequence[i], sequence[i+1])
    
    return G

def compute_spectrum(G):
    """Computes the spectrum (eigenvalues) of the Laplacian of the graph G."""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return np.array([0])  # Handle empty graphs
    
    L = nx.laplacian_matrix(G).todense()
    eigenvalues = np.linalg.eigvals(L)
    return np.sort(eigenvalues.real)  # Keep only real values

def save_graph_visualization(G, n, output_dir):
    """Saves a visualization of the graph G."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, node_size=100, node_color='lightblue', 
            with_labels=True, font_size=8, font_weight='bold')
    plt.title(f"Collatz Graph (n={n})")
    plt.savefig(os.path.join(output_dir, f"tree_{n:04d}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def precompute_graphs(n_max):
    """Precompute all graphs up to n_max in parallel."""
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        numbers = list(range(1, n_max + 1))
        graphs = list(tqdm(executor.map(build_collatz_graph, numbers), 
                           total=n_max, desc="Precomputing graphs"))
    return graphs

def animate_spectrum(n_max, save_as="collatz_spectrum.mp4"):
    """Creates an animated spectrum visualization and saves tree visualizations."""
    
    # Create output directory for tree visualizations
    trees_dir = "collatz_trees"
    os.makedirs(trees_dir, exist_ok=True)
    
    # Precompute all graphs
    graphs = precompute_graphs(n_max)
    
    # Save tree visualizations
    print("Saving tree visualizations...")
    for n, G in enumerate(tqdm(graphs), 1):
        save_graph_visualization(G, n, trees_dir)

    # Set up the figure
    fig, ax = plt.subplots()
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Evolution of the Collatz Graph Spectrum")
    line, = ax.plot([], [], 'o', markersize=3, markerfacecolor='blue', markeredgecolor='black', markeredgewidth=0.5)

    def init():
        """Initialize the animation."""
        line.set_data([], [])
        return line,

    def update(frame):
        """Update function for each frame."""
        n = frame + 1
        G = graphs[frame]
        eigenvalues = compute_spectrum(G)
        x_data = np.arange(len(eigenvalues))
        
        ax.clear()
        ax.set_xlim(-0.5, len(eigenvalues) - 0.5)
        ax.set_ylim(0, max(max(eigenvalues, default=0.1), 0.1))
        ax.plot(x_data, eigenvalues, 'o', markersize=3, markerfacecolor='blue', markeredgecolor='black', markeredgewidth=0.5)
        ax.set_title(f"Evolution of the Collatz Graph Spectrum (n={n})")
        ax.set_xlabel("Index")
        ax.set_ylabel("Eigenvalue")
        return line,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=n_max, init_func=init, repeat=False)

    # Save animation using FFMPEG with hardware acceleration
    writer = animation.FFMpegWriter(fps=30, codec="h264_videotoolbox", bitrate=5000)
    ani.save(save_as, writer=writer)
    print(f"Animation saved as {save_as}")
    print(f"Tree visualizations saved in {trees_dir}/")

if __name__ == "__main__":
    n_max = 100  # Adjust as needed
    animate_spectrum(n_max)