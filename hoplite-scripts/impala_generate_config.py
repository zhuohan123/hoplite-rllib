import yaml
import copy

with open("cartpole-impala-template.yaml", "r") as f:
    template = yaml.load(f, Loader=yaml.Loader)

print(template)

# for num_workers in [3, 7, 11, 15]:
for num_workers in [7, 15]:
    # for broadcast_interval in [2, 4, 6, 8]:
    broadcast_interval = (num_workers + 1) // 2
    for hoplite in [True, False]:
        config = copy.deepcopy(template)
        config["cartpole-impala"]["config"]["num_workers"] = num_workers
        config["cartpole-impala"]["config"]["broadcast_interval"] = broadcast_interval
        config["cartpole-impala"]["config"]["hoplite_config"]["enable"] = hoplite
        file_name = f"generated-cartpole-impala-gpu-{'hoplite' if hoplite else 'ray'}-{num_workers + 1}-{broadcast_interval}.yaml"
        with open(file_name, "w") as f:
            yaml.dump(config, f, Dumper=yaml.Dumper)

