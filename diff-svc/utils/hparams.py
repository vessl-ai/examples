import argparse
import os

import yaml

global_print_hparams = True
hparams = {}


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def override_config(old_config: dict, new_config: dict):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_config(old_config[k], new_config[k])
        else:
            old_config[k] = v


def set_hparams(
    config="",
    exp_name="",
    hparams_str="",
    print_hparams=True,
    global_hparams=True,
    reset=True,
    infer=True,
):
    """
    Load hparams from multiple sources:
    1. config chain (i.e. first load base_config, then load config);
    2. if reset == True, load from the (auto-saved) complete config file ('config.yaml')
       which contains all settings and do not rely on base_config;
    3. load from argument --hparams or hparams_str, as temporary modification.
    """
    if config == "":
        parser = argparse.ArgumentParser(description="set hparam")
        parser.add_argument(
            "--config", type=str, default="", help="location of the data corpus"
        )
        parser.add_argument("--reset", action="store_true", help="reset hparams")
        parser.add_argument("--exp_name", type=str, default="", help="exp_name")
        parser.add_argument(
            "--infer",
            help="whether preprocessing is for inference",
            action="store_true",
        )
        parser.add_argument(
            "--infer_ckpt_epoch",
            help="checkpoint epoch used for inference",
            type=int,
            default=0,
        )
        parser.add_argument("--org", help="VESSL organization", type=str, default="")
        parser.add_argument(
            "--vessl_project_name", help="VESSL project name", type=str, default=""
        )
        parser.add_argument(
            "--log_interval",
            help="interval for logging audio and checkpoints",
            type=int,
            default=50,
        )
        parser.add_argument(
            "--max_epoch",
            help="maximum epochs for training",
            type=int,
            default=1000,
        )
        args, unknown = parser.parse_known_args()
    else:
        args = Args(
            config=config,
            exp_name=exp_name,
            hparams=hparams_str,
            infer=infer,
            reset=reset,
        )
    args_work_dir = ""
    if args.exp_name != "":
        args.work_dir = args.exp_name
        args_work_dir = f"/input/vessl-diff-svc/checkpoints/{args.work_dir}"

    def load_config(config_fn):  # deep first
        with open(config_fn, encoding="utf-8") as f:
            hparams_ = yaml.safe_load(f)
        ret_hparams = hparams_
        return ret_hparams

    global hparams
    assert args.config != "" or args_work_dir != ""

    hparams_ = {}
    hparams_.update(load_config(args.config))

    hparams_["work_dir"] = args_work_dir
    ckpt_config_path = f"{args_work_dir}/config.yaml"

    if (
        args_work_dir != ""
        and (not os.path.exists(ckpt_config_path) or args.reset)
        and not args.infer
    ):
        os.makedirs(hparams_["work_dir"], exist_ok=True)
        with open(ckpt_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(hparams_, f)

    global global_print_hparams
    if global_hparams:
        hparams.clear()
        hparams.update(hparams_)

    # due to legacy configs, use infer_ckpt_epoch as key
    if config == "":
        hparams_["log_interval"] = args.log_interval
        hparams_["max_epoch"] = args.max_epoch
        hparams_["exp_name"] = args.exp_name
        hparams_["org"] = args.org
        hparams_["vessl_project_name"] = args.vessl_project_name
        hparams_["infer"] = args.infer
        hparams_["infer_ckpt_epoch"] = args.infer_ckpt_epoch
    hparams.update(hparams_)
    return hparams_
