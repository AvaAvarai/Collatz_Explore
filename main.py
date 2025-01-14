import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Function to compute the Collatz sequence
def collatz_sequence(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

# Function to generate a quadratic Bezier curve
def bezier_curve(p0, p1, p2, t):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

# Prepare the number line and Collatz sequences
numbers = list(range(1, 1001))  # Numbers from 1 to 1,000
sequences = {n: collatz_sequence(n) for n in numbers}

# Set up the plot
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 1000)  # Focus on the range 1–1,000
ax.set_ylim(0, 5)  # Narrower y-axis for bounce effect
ax.set_title("Collatz Conjecture: Animated Bouncing Curves for Numbers 1–1,000")
ax.set_xlabel("Number Line")
ax.set_ylabel("Bounce Height")
ax.grid(True)

# Initialize line artists for animation
lines = []
for _ in range(len(numbers)):
    line, = ax.plot([], [], lw=0.5, color="blue", alpha=0.5)
    lines.append(line)

# Animation function
def update_bouncing(frame):
    for i, number in enumerate(numbers):
        seq = sequences[number]
        if frame < len(seq) - 1:
            p0 = np.array([seq[frame], 0])  # Start point on the number line
            p1 = np.array([(seq[frame] + seq[frame + 1]) / 2, 10])  # Control point above
            p2 = np.array([seq[frame + 1], 0])  # End point on the number line
            t_vals = np.linspace(0, 1, 100)
            curve_x = bezier_curve(p0[0], p1[0], p2[0], t_vals)
            curve_y = bezier_curve(p0[1], p1[1], p2[1], t_vals)
            lines[i].set_data(curve_x, curve_y)
        else:
            lines[i].set_data([], [])  # Clear the line if the sequence is finished
    return lines

# Run the animation
ani_bounce = FuncAnimation(fig, update_bouncing, frames=200, interval=50, blit=True)

# Display the animation
plt.show()
