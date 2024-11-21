import numpy as np

max_inventory = 20
order_cost = 5
holding_cost_1 = 1
holding_cost_2 = 2
threshold = 1e-8
ITERATION = 100000

def limiting_distribution(states, n_states, state_to_idx):
    # Initialize the transition probability matrix
    P = np.zeros((n_states, n_states))

    # Calculate the transition matrix
    for s1, s2 in states:
        current_idx = state_to_idx[(s1, s2)]
        # Place order
        if s1 <= 1 or s2 <= 1:
            new_s1, new_s2 = 5, 5
            next_idx = state_to_idx[(new_s1, new_s2)]
            P[current_idx, next_idx] = 1 
        # No order placed
        else:
            for d1 in [0, 1]:
                for d2 in [0, 1]:
                    new_s1 = s1 - d1
                    new_s2 = s2 - d2
                    next_idx = state_to_idx[(new_s1, new_s2)]
                    P[current_idx, next_idx] += 0.25 # each combination P = 0.25

    pi = np.ones(n_states) / n_states

    for _ in range(ITERATION):
        new_pi = np.dot(pi, P)
        if np.max(np.abs(new_pi - pi)) < threshold:
            break
        pi = new_pi

    # Calculate long-run average costs
    ordering_cost = sum(
        pi[state_to_idx[(s1, s2)]] * ((s1 == 1) or (s2 == 1)) * order_cost
        for s1, s2 in states
    )
    holding_cost_total = sum(
        pi[state_to_idx[(s1, s2)]] * (s1 * holding_cost_1 + s2 * holding_cost_2)
        for s1, s2 in states
    )
    total_cost = ordering_cost + holding_cost_total
    return total_cost

def poisson_equation(states, n_states, state_to_idx):
    # Initialize the transition probability and reward matrix
    P = np.zeros((n_states, n_states))
    r = np.zeros(n_states)

    for s1, s2 in states:
        current_idx = state_to_idx[(s1, s2)]
        r[current_idx] = s1 * holding_cost_1 + s2 * holding_cost_2
        if s1 <= 1 or s2 <= 1:
            new_s1, new_s2 = 5, 5  # Both inventories reset to 5
            r[current_idx] += order_cost
            next_idx = state_to_idx[(new_s1, new_s2)]
            P[current_idx, next_idx] = 1  # Deterministic transition after ordering
        else:
            for d1 in [0, 1]:
                for d2 in [0, 1]:
                    new_s1 = s1 - d1
                    new_s2 = s2 - d2
                    next_idx = state_to_idx[(new_s1, new_s2)]
                    P[current_idx, next_idx] += 0.25
            

    # Initialize value function and average cost
    V = np.zeros(n_states)
    phi = 0
    
    # Calculate phi
    delta = threshold
    for _ in range(ITERATION):
        if delta < threshold:
            break
        V_new = np.zeros(n_states)
        for idx in range(n_states):
            # Poisson equation
            V_new[idx] = r[idx] + np.sum(P[idx, :] * V) - phi
        phi_new = np.mean(V_new)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        phi = phi_new

    return phi

def simulation():
    i1 = 20
    i2 = 20
    total_order_cost = 0
    total_holding_cost = 0

    for _ in range(ITERATION):
        order = False
        # Place order
        if i1 == 1 or i2 == 1:
            order = True
        # Decide demands
        d1 = np.random.choice([0, 1])
        d2 = np.random.choice([0, 1])
        # Sale
        i1 -= d1
        i2 -= d2
        # Order arrival
        if order:
            i1, i2 = 5, 5
            total_order_cost += order_cost
        # Calculate holding cost
        total_holding_cost += i1 * holding_cost_1 + i2 * holding_cost_2
    total_cost = total_order_cost + total_holding_cost
    return total_cost/ITERATION



if __name__ == "__main__":
    # State space: all combinations of inventory levels for product 1 and product 2
    states = [(s1, s2) for s1 in range(0, max_inventory + 1) for s2 in range(0, max_inventory + 1)]
    n_states = len(states)

    # Map states to indices
    state_to_idx = {state: idx for idx, state in enumerate(states)}

    simulation_result = simulation()
    limiting_distribution_result = limiting_distribution(states, n_states, state_to_idx)
    poisson_equation_result = poisson_equation(states, n_states, state_to_idx)

    print("Simulation: ", simulation_result)
    print("Limiting distribution: ", limiting_distribution_result)
    print("Poisson equation: ", poisson_equation_result)

