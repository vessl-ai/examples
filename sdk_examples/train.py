import vessl
import os
import time

access_token = os.environ.get("VESSL_ACCESS_TOKEN")

vessl.vessl_api.set_access_token(access_token)
vessl.vessl_api.set_organization("floyd")
vessl.vessl_api.set_project("vessl-mnist-examples")


def run(dataset, command, commit):
    experiment = vessl.create_experiment(
        cluster_name="aws-apne2-prod1",
        start_command=command,
        kernel_resource_spec_name="v1.cpu-2.mem-6",
        kernel_image_url="public.ecr.aws/vessl/kernels:py36.full-cpu",
        git_ref_mounts=[f"/root/examples:github/vessl-ai/examples/{commit}"],
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
    run(
        dataset="floyd/mnist-s3-test",
        command="pip install -r mnist/keras/requirements.txt && python mnist/keras/main.py --save-model --save-image",
        commit="master",
    )
