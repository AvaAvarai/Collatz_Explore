def collatz_cellular_automata(n):
    """
    Simulate the Collatz cellular automata for a given integer n.
    
    Args:
        n: A positive integer to start the simulation
    
    Returns:
        None (prints the cellular automata steps)
    """
    # Convert n to binary and remove '0b' prefix
    binary = bin(n)[2:]
    
    print(f"Starting with {n} (binary: {binary})")
    print("".join('1' if bit == '1' else '0' for bit in binary))
    
    # Continue until only one '1' remains
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

# Example usage
if __name__ == "__main__":
    try:
        n = int(input("Enter a positive integer: "))
        if n <= 0:
            print("Please enter a positive integer.")
        else:
            collatz_cellular_automata(n)
    except ValueError:
        print("Invalid input. Please enter a positive integer.")
