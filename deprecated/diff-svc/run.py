import importlib
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist
import vessl
from utils.hparams import hparams, set_hparams

set_hparams(print_hparams=False)
from infer import infer_on_target


def run_task():
    assert hparams["task_cls"] != ""
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task_cls.start()


if __name__ == "__main__":
    # configure VESSL
    organization_name = hparams["org"]
    vessl_project_name = hparams["vessl_project_name"]
    exp_name = hparams["exp_name"]

    vessl.configure(
        organization_name=organization_name, project_name=vessl_project_name
    )

    dataset_path = "/input/vessl-diff-svc"
    logging_dir = f"{dataset_path}/assets/logging"
    if not hparams["infer"]:
        # init ddp first
        default_port = 12910
        # if user gave a port number, use that one instead
        try:
            default_port = os.environ["MASTER_PORT"]
        except Exception:
            os.environ["MASTER_PORT"] = str(default_port)

        # figure out the root node addr
        def resolve_root_node_address(root_node):
            if "[" in root_node:
                name = root_node.split("[")[0]
                number = root_node.split(",")[0]
                if "-" in number:
                    number = number.split("-")[0]

                number = re.sub("[^0-9]", "", number)
                root_node = name + number

            return root_node

        root_node = "127.0.0.2"
        root_node = resolve_root_node_address(root_node)
        os.environ["MASTER_ADDR"] = root_node
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

        if local_rank == 0:
            vessl.init(message="model training")
            for file in os.listdir(logging_dir):
                audio_path = f"{logging_dir}/{file}"
                vessl.log(
                    payload={
                        "audio": [
                            vessl.Audio(audio_path, caption=f"original audio - {file}")
                        ]
                    }
                )

        run_task()
        # what happens in vessl managed experiments?
        if local_rank == 0:
            vessl.upload("/output/")
    else:
        # inference
        vessl.init(message="inference")
        exp_name = hparams["exp_name"]
        infer_ckpt_epoch = hparams["infer_ckpt_epoch"]
        infer_target_dir = f"/infer"
        ckpt_path = f"/ckpt/epoch_{infer_ckpt_epoch}.pt"
        out_dir = f"/output/{exp_name}"

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for file in os.listdir(infer_target_dir):
            file_path = f"{infer_target_dir}/{file}"
            out_path = f"{out_dir}/infer_epoch_{infer_ckpt_epoch}_{file}"
            infer_on_target(exp_name, ckpt_path, file_path, out_path)

            vessl.log(
                payload={
                    f"audio": [
                        vessl.Audio(file_path, caption=f"original audio for {file}")
                    ]
                }
            )

            vessl.log(
                payload={
                    f"audio": [
                        vessl.Audio(
                            out_path,
                            caption=f"inferred audio for {file} using model {ckpt_path}",
                        )
                    ]
                }
            )
