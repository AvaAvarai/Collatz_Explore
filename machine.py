def collatz_cellular_automata(n):
    """
    Simulate the Collatz cellular automata for a given integer n.
    
    Args:
        n: A positive integer to compute the simulation for numbers 1 to n
    
    Returns:
        None (prints the cellular automata steps and stopping time for each number)
    """
    ca_stopping_times = []
    collatz_stopping_times = []
    
    for i in range(1, n+1):
        # Calculate regular Collatz stopping time
        num = i
        collatz_steps = 0
        while num != 1:
            if num % 2 == 0:
                num = num // 2
            else:
                num = 3 * num + 1
            collatz_steps += 1
        collatz_stopping_times.append(collatz_steps)
        
        # Convert i to binary and remove '0b' prefix
        binary = bin(i)[2:]
        
        print(f"\nStarting with {i} (binary: {binary})")
        print("".join('1' if bit == '1' else '0' for bit in binary))
        
        # Continue until only one '1' remains
        steps = 0
        while binary.count('1') > 1:
            # Step 1: Append 1 to the end (2n + 1)
            binary_step1 = binary + '1'
            
            # Step 2: Add this to the original number (3n + 1)
            # Convert both to integers, add, then back to binary
            original_num = int(binary, 2)
            step1_num = int(binary_step1, 2)
            step2_num = original_num + step1_num
            binary_step2 = bin(step2_num)[2:]
            
            # Step 3: Remove all trailing 0s
            binary_step3 = binary_step2.rstrip('0')
            
            # Print the current state
            print("".join('1' if bit == '1' else '0' for bit in binary_step3))
            
            # Update binary for next iteration
            binary = binary_step3
            
            # Increment step counter
            steps += 1
        
        print(f"Stopping time for {i}: {steps} steps (CA), {collatz_steps} steps (Collatz)")
        ca_stopping_times.append(steps)
    
    # Plot the stopping times as a line graph
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n+1), ca_stopping_times, marker='o', linestyle='-', label='Cellular Automata')
        plt.plot(range(1, n+1), collatz_stopping_times, marker='x', linestyle='--', label='Collatz Function')
        plt.title('Collatz vs Cellular Automata Stopping Times')
        plt.xlabel('Starting Number')
        plt.ylabel('Stopping Time (steps)')
        plt.legend()
        plt.grid(True)
        plt.savefig('collatz_stopping_times.png')
        plt.show()
        print(f"Graph saved as 'collatz_stopping_times.png'")
    except ImportError:
        print("Matplotlib not installed. Cannot create plot.")


if __name__ == "__main__":
    try:
        n = int(input("Enter a positive integer n to compute for numbers 1 to n: "))
        if n <= 0:
            print("Please enter a positive integer.")
        else:
            collatz_cellular_automata(n)
    except ValueError:
        print("Invalid input. Please enter a positive integer.")
