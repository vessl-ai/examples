import argparse

import torch
import yaml


def main(layer_cls_name):
    with open("configs/fsdp_config.yaml") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    config["num_processes"] = torch.cuda.device_count()
    config["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"] = layer_cls_name
    with open("configs/fsdp_config.yaml", "w") as fp:
        yaml.dump(config, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-cls-name", required=True)
    args = parser.parse_args()

    main(args.layer_cls_name)
