import duckdb
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

print (sys.argv)
pair_range = [int(sys.argv[1]), int(sys.argv[2])]
scale = int(sys.argv[3])
xticks = list(filter(lambda value: value >= pair_range[0] and value <= pair_range[1], [1, 10, 100, 1000, 2048, 4096, 8192, 16834, 32768, 65536]))

def split_first_word(string):
    return string.split(' ', 1)
ind = np.arange(len(xticks))
width = 0.2
experiments = [split_first_word(x) for x in sys.argv[4:]]
# ['scale out', 'scale out scale for', 'scale out scale up', 'scale out scale up parallel for']

with duckdb.connect("results.db") as con:
    for x, experiment in enumerate(experiments):
        sql = f"SELECT * FROM experiment_results WHERE experiment='{experiment[1]}' AND backend='{experiment[0]}' AND scale = {scale} AND pairs >= {pair_range[0]} AND pairs <= {pair_range[1]};"
        result = con.sql(sql).df()
        plt.bar(ind + width * x, result['wall_time'].to_numpy(), width)

plt.xticks(ind + (width * (len(experiments) - 1) / 2 ), xticks)
plt.legend(sys.argv[4:])
plt.xlabel("pairs")
plt.ylabel("time in seconds")
plt.title(f"Scale factor {scale}")
plt.savefig("output.pdf")
