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
    if log_file.startswith("generated-cartpole-a3c") and file_name[1] == "yaml-latest":
        config = file_name[0].split("-")
        with open(os.path.join(log_dir, log_file), "r") as f:
            n_nodes = int(config[5])
            b_interval = int(config[6])
            results = []
            num_steps_trained = []
            for line in f:
                if "time_this_iter_s" in line:
                    results.append(float(line.split()[1]))
                if "num_steps_trained" in line:
                    num_steps_trained.append(int(line.split()[1]))
            for i in reversed(range(1, len(num_steps_trained))):
                num_steps_trained[i] -= num_steps_trained[i - 1]
            s = np.array(num_steps_trained)[1:11] / np.array(results)[1:11]
            all_results.append((n_nodes, b_interval, config[4], s.mean(), s.std()))

all_results = sorted(all_results)
for r in all_results:
    print("\t".join([str(x) for x in r]))
with open(os.path.join(log_dir, "results.json"), "w") as f:
    json.dump(all_results, f)
