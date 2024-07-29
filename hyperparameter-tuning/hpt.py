import itertools
import os

import vessl

from jinja2 import Environment, FileSystemLoader


dir_path = os.path.dirname(os.path.realpath(__file__))
file_loader = FileSystemLoader(dir_path)
env = Environment(loader=file_loader)

template = env.get_template("run_template.yaml")

parameters = {
    "lr": [0.1, 0.01],
    "batch_size": [64, 128],
}

keys = parameters.keys()
combinations = itertools.product(*[parameters[key] for key in keys])

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
