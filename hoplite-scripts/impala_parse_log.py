import sys
import os
import numpy as np
import json

log_dir = 'log/'
if len(sys.argv) > 1:
    log_dir = sys.argv[1]
print("log_dir:", log_dir)

all_results = []

for log_file in os.listdir(log_dir):
    file_name = log_file.split(".")
    if log_file.startswith("generated") and file_name[1] == "yaml-latest":
        config = file_name[0].split("-")
        with open(os.path.join(log_dir, log_file), "r") as f:
            results = []
            for line in f:
                if "train_throughput" in line:
                    results.append(float(line.split()[1]))
            s = np.array(results[3: -3])
            all_results.append((int(config[5]), int(config[6]), config[4], s.mean(), s.std()))

all_results = sorted(all_results)
for r in all_results:
    print("\t".join([str(x) for x in r]))
with open(os.path.join(log_dir, "results.json"), "w") as f:
    json.dump(all_results, f)
