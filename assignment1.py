import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import time


def dp_normal(T, initial_inventory, prices, probabilities):
    # V[t][i] is the expected maximal reward for state (t,i)
    V = np.zeros((T + 1, initial_inventory + 1))
    # Optimal policy
    policy = np.zeros((T, initial_inventory + 1), dtype=int)

    # Iterate over all time periods
    for t in range(T - 1, -1, -1):
        # Iterate over all inventory levels
        for i in range(initial_inventory + 1):
            max_value = -np.inf
            best_action = None

            # Evaluate each price option
            for price_index, (price, prob) in enumerate(zip(prices, probabilities)):
                expected_reward = price * prob

                # If inventory > 0, calculate state transition
                if i > 0:
                    future_value = prob * V[t + 1][i - 1] + (1 - prob) * V[t + 1][i]
                else:
                    future_value = 0

                total_value = expected_reward + future_value

                # Update the max reward and the best solution
                if total_value > max_value:
                    max_value = total_value
                    best_action = price_index

            # Store the max reward and the best solution
            V[t][i] = max_value
            policy[t][i] = best_action

    # Report the expected maximal reward
    max_expected_reward = V[0][initial_inventory]
    print(f"Expected Maximal Reward: {max_expected_reward:.2f}")
    return policy


def simulation_normal(policy):
    num_simulations = 1000
    total_revenues = []

    for _ in range(num_simulations):
        revenue = 0
        inventory = initial_inventory

        for t in range(T):
            if inventory == 0:
                break

            # Get the optimal action for the current time and inventory
            price_index = policy[t][inventory]
            price = prices[price_index]
            probability_of_sale = probabilities[price_index]

            # Randomly decide if a sale happens
            sale_happens = np.random.rand() < probability_of_sale
            if sale_happens:
                revenue += price
                inventory -= 1
        total_revenues.append(revenue)

    average_reward = np.mean(total_revenues)
    print(f"Average Reward from 1000 simulations: {average_reward:.2f}")
    return total_revenues


def diagrams(policy, total_revenues, prices):
    # Map policy indices to actual prices
    optimal_policy = np.vectorize(lambda a: prices[a])(policy)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Ensure prices and colors are in the same order
    # Sort prices and get corresponding indices
    sorted_prices_indices = np.argsort(prices)
    sorted_prices = np.array(prices)[sorted_prices_indices]
    sorted_colors = ['#1D3557', '#457B9D', '#A8DADC']  # Dark to light blues

    # Create colormap with colors corresponding to sorted prices
    cmap = ListedColormap(sorted_colors)

    # Define boundaries for the colormap normalization
    # Boundaries need to be one more than the number of colors
    boundaries = np.append(sorted_prices - 25, sorted_prices[-1] + 25)
    norm = BoundaryNorm(boundaries, cmap.N)

    # Plot the optimal pricing policy with distinct colors for each price level
    cax = axes[0].pcolormesh(optimal_policy.T, cmap=cmap, norm=norm, shading='auto')
    axes[0].set_xlabel('Time Period')
    axes[0].set_ylabel('Inventory')
    axes[0].set_title('Optimal Pricing Policy')

    # Adding color bar with labels for each price level
    cbar = fig.colorbar(cax, ax=axes[0], ticks=sorted_prices)
    cbar.set_label('Price')
    cbar.ax.set_yticklabels([f'${price}' for price in sorted_prices])

    # Histogram of total revenues
    axes[1].hist(total_revenues, bins=30, edgecolor='k', alpha=0.7)
    axes[1].set_xlabel('Total Revenue')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Total Revenues')

    plt.tight_layout()
    plt.show()

    fig.savefig('/Users/jackdaw/Downloads/assignment1_1.png')


def dp_with_constraint(T, initial_inventory, prices, probabilities):
    # V[t][i][a]
    V = np.zeros((T + 1, initial_inventory + 1, len(prices)))
    policy = np.zeros((T, initial_inventory + 1, len(prices)), dtype=int)

    for t in range(T - 1, -1, -1):
        for x in range(initial_inventory + 1):  # for each inventory level
            for price_index, p in enumerate(prices):  # for each max allowed price level
                max_value = -np.inf
                best_action = None

                # Evaluate each price option <= p
                for a_index, (price, prob) in enumerate(zip(prices, probabilities)):
                    if price > p:  # Skip if price exceeds max allowed price
                        continue

                    expected_reward = price * prob

                    # If inventory > 0, calculate state transition
                    if x > 0:
                        future_value = prob * V[t + 1][x - 1][a_index] + (1 - prob) * V[t + 1][x][a_index]
                    else:
                        future_value = 0

                    total_value = expected_reward + future_value

                    # Update max reward and best action
                    if total_value > max_value:
                        max_value = total_value
                        best_action = a_index

                V[t][x][price_index] = max_value
                policy[t][x][price_index] = best_action

    max_expected_reward = V[0][initial_inventory][0]
    print(f"Expected Maximal Reward with Constraint: {max_expected_reward:.2f}")
    return policy


def heatmap_constraint(constraint_policy, prices):
    # Assume the initial previous price is the highest price
    initial_prev_price_index = 0

    # Extract the optimal policy for the initial previous price
    # constraint_policy has shape (T, inventory_levels, price_levels)
    optimal_policy_indices = constraint_policy[:, :, initial_prev_price_index]

    # Map policy indices to actual prices
    optimal_policy = np.vectorize(lambda a: prices[a])(optimal_policy_indices)
    # Define a colormap with distinct colors for each price level

    sorted_prices = sorted(prices)
    boundaries = sorted_prices + [sorted_prices[-1] + 50]

    cmap = ListedColormap(['#1D3557', '#457B9D', '#A8DADC'])
    norm = BoundaryNorm(boundaries, cmap.N)

    # Plot the optimal pricing policy using imshow
    plt.figure(figsize=(12, 8))
    plt.imshow(optimal_policy.T, aspect='auto', cmap=cmap, norm=norm, origin='lower')

    plt.xlabel('Time Period')
    plt.ylabel('Inventory Level')
    plt.title('Optimal Pricing Policy under Constraint of No Price Increases')

    # Create a colorbar with labels corresponding to the prices
    cbar = plt.colorbar(ticks=prices)
    cbar.set_label('Price')
    cbar.ax.set_yticklabels([f'${price}' for price in prices])

    plt.tight_layout()
    plt.savefig('/Users/jackdaw/Downloads/assignment1_2.png')
    plt.show()



    # # Map policy indices to actual prices and ensure it's 2D
    # optimal_policy = np.vectorize(lambda a: prices[a])(constraint_policy)
    # optimal_policy = np.squeeze(optimal_policy)  # Remove any extra dimensions if necessary
    #
    # fig, ax = plt.subplots(figsize=(15, 5))
    #
    # # Sort prices and define colormap
    # sorted_prices_indices = np.argsort(prices)
    # sorted_prices = np.array(prices)[sorted_prices_indices]
    # sorted_colors = ['#1D3557', '#457B9D', '#A8DADC']  # Dark to light blues
    #
    # cmap = ListedColormap(sorted_colors)
    # boundaries = np.append(sorted_prices - 25, sorted_prices[-1] + 25)
    # norm = BoundaryNorm(boundaries, cmap.N)
    #
    # # Plot the optimal pricing policy with distinct colors for each price level
    # cax = ax.pcolormesh(optimal_policy.T, cmap=cmap, norm=norm, shading='auto')
    # ax.set_xlabel('Time Period')
    # ax.set_ylabel('Inventory')
    # ax.set_title('Optimal Pricing Policy')
    #
    # # Adding color bar with labels for each price level
    # cbar = fig.colorbar(cax, ax=ax, ticks=sorted_prices)
    # cbar.set_label('Price')
    # cbar.ax.set_yticklabels([f'${price}' for price in sorted_prices])
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    T = 500
    initial_inventory = 100

    prices = [200, 100, 50]
    probabilities = [0.1, 0.5, 0.8]

    # Normal policy
    start_time = time.time()
    normal_policy = dp_normal(T, initial_inventory, prices, probabilities)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Execution time for Normal policy: {runtime:.4f} seconds")

    # Simulation Over 1000 runs
    normal_revenue = simulation_normal(normal_policy)

    # Diagrams for Normal policy and Simulation
    diagrams(normal_policy, normal_revenue, prices)

    # Policy with Constraint
    constraint_policy = dp_with_constraint(T, initial_inventory, prices, probabilities)
    heatmap_constraint(constraint_policy, prices)
