def collatz_abstract_machine(n):
    """
    Simulate the Collatz abstract machine for a given integer n.
    
    This function implements a string rewriting system that follows these rules:
    1. Convert the number to binary representation
    2. For each step of the abstract machine:
       a. Append '1' to the binary string (equivalent to 2n + 1)
       b. Add this new number to the original number (producing 3n + 1)
       c. Remove all trailing zeros from the result
    3. Continue until only a single '1' remains in the binary string
    
    The stopping time is the number of steps required to reach a string with only one '1'.
    
    Args:
        n: A positive integer to compute the simulation for numbers 1 to n
    
    Returns:
        None (saves the abstract machine steps and stopping time for each number to a timestamped file)
    """
    import datetime
    
    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"collatz_am_results_{timestamp}.txt"
    
    am_stopping_times = []
    collatz_stopping_times = []
    
    with open(filename, 'w') as f:
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
            
            f.write(f"\nStarting with {i} (binary: {binary})\n")
            f.write("".join('1' if bit == '1' else '0' for bit in binary) + "\n")
            
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
                
                # Write the current state
                f.write("".join('1' if bit == '1' else '0' for bit in binary_step3) + "\n")
                
                # Update binary for next iteration
                binary = binary_step3
                
                # Increment step counter
                steps += 1
            
            f.write(f"Stopping time for {i}: {steps} steps (AM), {collatz_steps} steps (Collatz)\n")
            am_stopping_times.append(steps)
    
    print(f"Results saved to {filename}")
    
    # Plot the stopping times as a line graph
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n+1), collatz_stopping_times, marker='x', linestyle='--', label='Collatz Function')
        plt.plot(range(1, n+1), am_stopping_times, marker='o', linestyle='-', label='Abstract Machine')
        plt.title('Collatz vs Abstract Machine Stopping Times')
        plt.xlabel('Starting Number')
        plt.ylabel('Stopping Time (steps)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'collatz_stopping_times_{timestamp}.png')
        print(f"Graph saved as 'collatz_stopping_times_{timestamp}.png'")
    except ImportError:
        print("Matplotlib not installed. Cannot create plot.")


if __name__ == "__main__":
    try:
        n = int(input("Enter a positive integer n to compute for numbers 1 to n: "))
        if n <= 0:
            print("Please enter a positive integer.")
        else:
            collatz_abstract_machine(n)
    except ValueError:
        print("Invalid input. Please enter a positive integer.")
