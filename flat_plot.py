import matplotlib.pyplot as plt
import random

def collatz_sequence(n):
    """Generate the Collatz sequence starting from n."""
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

def plot_collatz_all(max_n):
    """Plot all Collatz sequences for numbers from 1 to max_n on one graph."""
    plt.figure(figsize=(12, 8))
    for n in range(1, max_n + 1):
        sequence = collatz_sequence(n)
        plt.plot(sequence, label=f"n={n}")
    plt.title(f"Collatz Conjecture Sequences for n=1 to {max_n}")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.show()

# Example: Plot all sequences for n up to 100
plot_collatz_all(100)
