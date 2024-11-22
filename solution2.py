import numpy as np
import random

def knapsack_gwo(values, weights, capacity, population_size=10, max_iter=100):
    """
    Solve the knapsack problem using the Grey Wolf Optimization algorithm.
    Args:
        values (list): Values of the items.
        weights (list): Weights of the items.
        capacity (int): The maximum weight the knapsack can carry.
        population_size (int): Number of wolves in the population.
        max_iter (int): Maximum number of iterations.
    Returns:
        list: The best solution found (binary list representing selected items).
        int: The maximum value achieved.
    """
    num_items = len(values)

    # Initialize the wolf population randomly
    def initialize_population():
        population = []
        for _ in range(population_size):
            wolf = [random.randint(0, 1) for _ in range(num_items)]
            while np.dot(wolf, weights) > capacity:
                wolf = [random.randint(0, 1) for _ in range(num_items)]
            population.append(wolf)
        return np.array(population)

    population = initialize_population()

    # Fitness function
    def fitness(wolf):
        total_weight = np.dot(wolf, weights)
        total_value = np.dot(wolf, values)
        return total_value if total_weight <= capacity else 0

    # Identify the alpha, beta, and delta wolves
    def get_leaders(population):
        fitness_values = [fitness(wolf) for wolf in population]
        sorted_indices = np.argsort(fitness_values)[::-1]
        return population[sorted_indices[0]], population[sorted_indices[1]], population[sorted_indices[2]]

    alpha, beta, delta = get_leaders(population)

    # Update positions of wolves
    def update_position(wolf, a):
        new_wolf = []
        for i in range(num_items):
            r1, r2 = random.random(), random.random()
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = abs(C1 * alpha[i] - wolf[i])
            X1 = alpha[i] - A1 * D_alpha

            r3, r4 = random.random(), random.random()
            A2, C2 = 2 * a * r3 - a, 2 * r4
            D_beta = abs(C2 * beta[i] - wolf[i])
            X2 = beta[i] - A2 * D_beta

            r5, r6 = random.random(), random.random()
            A3, C3 = 2 * a * r5 - a, 2 * r6
            D_delta = abs(C3 * delta[i] - wolf[i])
            X3 = delta[i] - A3 * D_delta

            X_new = (X1 + X2 + X3) / 3
            new_wolf.append(1 if random.random() < X_new else 0)

        while np.dot(new_wolf, weights) > capacity:
            new_wolf = [random.randint(0, 1) for _ in range(num_items)]
        return new_wolf

    # Main loop of the GWO algorithm
    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)
        for i in range(population_size):
            population[i] = update_position(population[i], a)

        alpha, beta, delta = get_leaders(population)

    best_solution = alpha
    best_value = fitness(alpha)
    return best_solution, best_value


# Test with the same values as the dynamic programming example
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50

# Solve the problem using GWO
best_solution_gwo, max_value_gwo = knapsack_gwo(values, weights, capacity)

# Print the results
print("Best solution (items selected):", best_solution_gwo)
print("Maximum value achieved:", max_value_gwo)