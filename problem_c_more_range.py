import matplotlib.pyplot as plt

# Calculate steps for different ranges
ranges = [100, 1000, 10000]
plt.figure(figsize=(15, 5))

for i, max_num in enumerate(ranges, 1):
    steps = []
    for x in range(1, max_num + 1):
        n = x
        count = 0
        while n != 1:
            if n % 2 == 0:
                n //= 2
            else:
                n = 3 * n + 1
            count += 1
        steps.append(count)

    # Create subplot for each range
    plt.subplot(1, 3, i)
    plt.plot(range(1, max_num + 1), steps, 'b-', linewidth=0.5)
    plt.grid(True)
    plt.xlabel('Starting Number (x)')
    plt.ylabel('Number of Steps to Reach 1')
    plt.title(f'Steps to Reach 1 for Numbers 1-{max_num}')

plt.tight_layout()
plt.show()
