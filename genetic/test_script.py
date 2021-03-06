"""
This is the test script for optimizers in the genetic package.
"""
import sys
from typing import Dict

import numpy as np
# from matplotlib import pyplot as plt

sys.path.append("./")
from optimizer import GeneticOptimizer
from genetic.tuner import GeneticTuner


def obj_func(param: Dict[str, object]) -> float:
    # The example problem
    # Soln: (x, y) = (-3, 2)
    # Minimal: 6
    x, y = param.values()
    f = (x+3)**2 + (y-2)**2 + 6
    return f


# Searching range
lb = 2e10
ub = -2e10
init_size = 500
epochs = 100

candidates = (ub - lb) * np.random.random(init_size) + lb

gene_pool = {
    "x": list(candidates),
    "y": list(candidates)
    }

# Sample chromosomes.
f1 = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
f2 = {"a": [7.0, 8.0, 9.0], "b": [10.0, 11.0, 12.0]}
f3 = {"a": [13.0, 14.0], "b": [15.0, 16.0]}

i1 = {"a": [1, 2, 3], "b": [4, 5, 6]}
i2 = {"a": [7, 8, 9], "b": [10, 11, 12]}
i3 = {"a": [13, 14], "b": [15, 16]}

optimizer = GeneticOptimizer(
    gene_pool=gene_pool,
    pop_size=init_size,
    eval_func=obj_func,
    mode="max",
    retain=0.5,
    shot_prob=0.05,
    mutate_prob=0.05,
    verbose=False
)

# (a, b) = optimizer.cross_over(f1, i3)
# print(a)
# print(b)

optimizer.mutate(i2, mutate_prob=1.0)

optimizer.evaluate(verbose=True)
for e in range(epochs):
    print(f"Generation: [{e+1}/{epochs}]")
    optimizer.select()
    # print(optimizer.count_population())
    optimizer.evolve()
    optimizer.evaluate(verbose=True)

optimizer.count_population()
sum(isinstance(x, tuple) for x in optimizer.population)

print(f"Optimizer x-star found at {optimizer.population[0][0]}")
print(f"extremal value attained: {obj_func(optimizer.population[0][0]):0.5f}")

# print("More attentions are required if the maximizer/minimizer is near boundary.")
