import yaml
import copy

with open("cartpole-a3c-template.yaml", "r") as f:
    template = yaml.load(f, Loader=yaml.Loader)

print(template)

# for num_workers in [3, 7, 11, 15]:
for num_workers in [7, 15]:
    # for broadcast_interval in [2, 4, 6, 8]:
    broadcast_interval = (num_workers + 1) // 2
    for hoplite in [True, False]:
        config = copy.deepcopy(template)
        config["cartpole-a3c"]["config"]["num_workers"] = num_workers
        config["cartpole-a3c"]["config"]["optimizer"]["broadcast_interval"] = broadcast_interval
        config["cartpole-a3c"]["config"]["hoplite_config"]["enable"] = hoplite
        config["cartpole-a3c"]["config"]["optimizer"]["use_hoplite"] = hoplite
        file_name = f"generated-cartpole-a3c-gpu-{'hoplite' if hoplite else 'ray'}-{num_workers + 1}-{broadcast_interval}.yaml"
        with open(file_name, "w") as f:
            yaml.dump(config, f, Dumper=yaml.Dumper)

