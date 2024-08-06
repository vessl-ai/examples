import itertools
import os
import random

import vessl

from jinja2 import Environment, FileSystemLoader


dir_path = os.path.dirname(os.path.realpath(__file__))
file_loader = FileSystemLoader(dir_path)
env = Environment(loader=file_loader)

template = env.get_template("run_template.yaml")

# grid search
batch_size_min = 64
batch_size_max = 128
batch_size_step = 64

batch_size = range(batch_size_min, batch_size_max + batch_size_step, batch_size_step)

lr = [0.01, 0.0001]

parameters = {
    "lr": lr,
    "batch_size": batch_size,
}

keys = parameters.keys()
combinations = itertools.product(*[parameters[key] for key in keys])

#random_search
seed = 42
random.seed(seed)

batch_size_values = [64, 128, 256]

lr_min = 0.1
lr_max = 0.2

combinations_count = 4

combinations = [
    (random.choice(batch_size_values), random.uniform(lr_min, lr_max)) for _ in range(combinations_count)
]

for c in combinations:
    data = dict()
    for i, key in enumerate(keys):
        data[key] = c[i]
    run = template.render(data)

    vessl.create_run(
        yaml_file="",
        yaml_body=run,
        yaml_file_name="hpt.yaml"
    )
