import numpy as np

steps = 10000
holding_cost_1 = 1
holding_cost_2 = 2
order_cost = 5

i1 = 20
i2 = 20

total_order_cost = 0
total_holding_cost = 0

for step in range(steps):
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
    # print(i1, i2, d1, d2)

    # Order arrival
    if order:
        i1, i2 = 5, 5
        total_order_cost += order_cost
    
    # Calculate holding cost
    total_holding_cost += i1 * holding_cost_1 + i2 * holding_cost_2
    # print("end day inventory: ", i1, i2)

total_cost = total_order_cost + total_holding_cost
print("Avg cost: ", total_cost/steps)

