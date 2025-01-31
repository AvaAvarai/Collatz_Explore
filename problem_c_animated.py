import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(1, 100)
ax.grid(True)
ax.set_xlabel('Starting Number (x)')
ax.set_ylabel('Number of Steps to Reach 1')
ax.set_title('Steps to Reach 1 for Numbers 1-100')

# Initialize empty line
line, = ax.plot([], [], 'b-', linewidth=0.5)

# Animation initialization function
def init():
    line.set_data([], [])
    return line,

# Animation update function
def update(frame):
    steps = []
    for x in range(1, frame + 2):
        n = x
        count = 0
        while n != 1:
            if n % 2 == 0:
                n //= 2
            else:
                n = 3 * n + 1
            count += 1
        steps.append(count)
    
    line.set_data(range(1, frame + 2), steps)
    ax.set_ylim(0, max(steps) + 1)
    return line,

# Create animation
anim = FuncAnimation(fig, update, frames=99, init_func=init,
                    interval=50, blit=True)

# Save as GIF
anim.save('collatz_steps.gif', writer='pillow')
plt.close()
