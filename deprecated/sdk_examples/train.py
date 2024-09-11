import os
import time

import vessl

access_token = os.environ.get("VESSL_ACCESS_TOKEN")

vessl.vessl_api.set_access_token(access_token)
vessl.vessl_api.set_organization("jaeman")
vessl.vessl_api.set_project("examples2")


def run_with_local_file_upload(dataset, command):
    experiment = vessl.create_experiment(
        cluster_name="aws-apne2-prod1",
        start_command=command,
        kernel_resource_spec_name="v1.cpu-2.mem-6",
        kernel_image_url="public.ecr.aws/vessl/kernels:py36.full-cpu",
        local_files=["/Users/kuss/Workspace/savvi/examples/:/root/local/"],
        dataset_mounts=[f"/input/mnist:{dataset}"],
        working_dir="/root/local",
    )
    print(experiment)


def run_with_custom_resource(dataset, command):
    experiment = vessl.create_experiment(
        cluster_name="vessl-dgx",
        start_command=command,
        processor_type="GPU",
        cpu_limit=2,
        memory_limit="4Gi",
        gpu_type="Any",
        gpu_limit=1,
        kernel_image_url="public.ecr.aws/vessl/kernels:py36.full-gpu",
        dataset_mounts=[f"/input/mnist:{dataset}"],
        local_files=["/Users/kuss/Workspace/savvi/examples/:/root/local/"],
        working_dir="/root/local/",
    )
    print(experiment)


def run(dataset, command, commit):
    experiment = vessl.create_experiment(
        cluster_name="aws-apne2-prod1",
        start_command=command,
        kernel_resource_spec_name="v1.cpu-2.mem-6",
        kernel_image_url="public.ecr.aws/vessl/kernels:py36.full-cpu",
        git_ref_mounts=[f"/root/examples:github/support-vessl/examples/{commit}"],
        dataset_mounts=[f"/input/mnist:{dataset}"],
        working_dir="/root/examples",
    )
    while True:
        time.sleep(5)
        experiment = vessl.read_experiment(experiment.number)
        logs = vessl.list_experiment_logs(experiment.number, 1)
        message = ""
        if logs:
            message = f"{logs[0].message} {logs[0].timestamp}"
        print(experiment.number, experiment.status, message)

        if experiment.status in ["completed", "terminated", "failed"]:
            break


if __name__ == "__main__":
    run_with_custom_resource(
        dataset="floyd/mnist-s3-test",
        command="pip install -r mnist/keras/requirements.txt && python mnist/keras/main.py --save-model --save-image",
    )
    # run_with_local_file_upload(
    #    dataset="floyd/mnist-s3-test",
    #    command="pip install -r mnist/keras/requirements.txt && python mnist/keras/main.py --save-model --save-image",
    # )
    # run(
    #     dataset="floyd/mnist-s3-test",
    #     command="pip install -r mnist/keras/requirements.txt && python mnist/keras/main.py --save-model --save-image",
    #     commit="master",
    # )
