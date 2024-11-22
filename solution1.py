def knapsack_dynamic_programming(values, weights, capacity):
    """
    Solve the knapsack problem using dynamic programming.
    Args:
        values (list): The value of each item.
        weights (list): The weight of each item.
        capacity (int): The maximum weight the knapsack can carry.
    Returns:
        int: The maximum value that can be achieved.
    """
    # Number of items
    n = len(values)

    # Step 1: Create a table to store the maximum value for each weight limit
    # dp[i][w] will store the maximum value using the first 'i' items and a weight limit 'w'
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Step 2: Fill the dp table
    for i in range(1, n + 1):  # Loop through each item
        for w in range(1, capacity + 1):  # Loop through each weight limit
            if weights[i - 1] <= w:  # Check if the item's weight fits in the current weight limit
                # Option 1: Include the item
                include_value = values[i - 1] + dp[i - 1][w - weights[i - 1]]
                # Option 2: Exclude the item
                exclude_value = dp[i - 1][w]
                # Take the maximum of the two options
                dp[i][w] = max(include_value, exclude_value)
            else:
                # If the item doesn't fit, just carry forward the value without it
                dp[i][w] = dp[i - 1][w]

    # Step 3: The maximum value for the full weight limit is in dp[n][capacity]
    return dp[n][capacity]


# Example Usage:
values = [60, 100, 120]  # Values of the items
weights = [10, 20, 30]  # Weights of the items
capacity = 50  # Maximum weight the knapsack can carry

# Solve the problem
max_value = knapsack_dynamic_programming(values, weights, capacity)

# Print the result
print("The maximum value we can achieve is:", max_value)