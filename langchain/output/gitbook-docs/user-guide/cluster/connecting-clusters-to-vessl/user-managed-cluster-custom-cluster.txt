# User-managed cluster (Custom cluster)

VESSL support cluster management features for users' Kubernetes clusters. You can easily register your own Kubernetes cluster to VESSL whether it's on-premise, on-cloud or a mix of ones.

## Registering custom cluster to VESSL

### 1. Get Helm CLI to Install VESSL Cluster Agent

The VESSL Cluster Agent provides centralized approach for managing and monitoring ML workloads in the cluster. Using VESSL Cluster Agent enables you to:

* Connect to VESSL API server to run VESSL-managed workloads on the cluster
* Monitor issues and system metrics for cluster, nodes and workloads on VESSL
* Keep track of resource usage of the cluster to achieve resource governance

VESSL uses [Helm](https://helm.sh), the package manager for Kubernetes, to install the cluster agent. Install Helm CLI on your local machine to run VESSL's agent install command.

{% tabs %}
{% tab title="Shell Script" %}
```
curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash
```
{% endtab %}

{% tab title="Homebrew" %}
```
brew install helm
```
{% endtab %}
{% endtabs %}

### 2. Get Agent Install Command from VESSL

Select **"on-premise"** from the **"Type"** section on the new cluster dialog. Additional form inputs will show up to set cluster name and Kubernetes namespace to install VESSL agent. Set cluster name and Kubernetes namespace, and click **"Next"** button.

![](<../../../.gitbook/assets/image (170).png>)

A Helm command to install cluster agent will show up on the next step. While using the right [Kubectl context](https://kubernetes.io/docs/tasks/access-application-cluster/configure-access-multiple-clusters/#define-clusters-users-and-contexts) pointing your cluster, run the command to install cluster agent.

![](<../../../.gitbook/assets/Screen Shot 2022-02-25 at 2.09.41 PM.png>)

Once installed, cluster agent will try to authenticate and connect to VESSL API server. VESSL will let you know whether the agent is successfully connected to VESSL.

![](<../../../.gitbook/assets/image (148).png>)

If cluster connection status is not changing to 'success' for a couple of minutes, check logs from the cluster agent to troubleshoot the issue.



