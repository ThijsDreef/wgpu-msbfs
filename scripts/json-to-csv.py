import duckdb
import re
import json

var = "wgpu_coalesced_access.json"

experiment = "coalesced access"
device = "personal laptop"
gpu = "NVIDIA 2080 RTX MAX Q design"
backend = "wgpu"

with duckdb.connect("results.db") as con:
    with open(f"{var}", 'r') as file:
        data = json.load(file)

    for test in data['benchmarks']:
        match = re.search('Scale(.*)Pairs(.*)', test['name'])
        scale = match.group(1)
        pairs = match.group(2)

        sql = f"DELETE FROM experiment_results WHERE experiment = '{experiment}' AND backend='{backend}' AND device = '{device}' AND gpu = '{gpu}' AND pairs = {int(pairs)} AND scale = {int(scale)}"
        print(sql)
        con.sql(sql)
        sql = f"INSERT INTO experiment_results VALUES ('{experiment}', '{device}', '{gpu}', '{backend}', {int(pairs)}, {int(scale)}, {float(test['cpu_time'])}, {float(test['real_time'])})"
        print(sql)
        con.sql(sql)


    con.table("experiment_results").show()
