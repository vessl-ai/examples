---
description: Adding custom clusters
---

# Configure Clusters

### Bring your own clusters

You can easily register your own custom cluster to VESSL whether it's on-premise, on-cloud or a combination of the two.

### Adding custom cluster to VESSL

#### 1. Install Helm

VESSL uses [Helm](https://helm.sh), the package manager of Kubernetes. Begin by installing Helm.&#x20;

{% tabs %}
{% tab title="Script" %}
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

**2. Helm Install Kubernetes resources**

To add Kubernetes cluster to VESSL, you should first create the followings in the cluster:

* A service account that has all the permissions in a namespace
* VESSL cluster agent

You can see the preview of result on [VESSL's GitHub repository](https://github.com/vessl-ai/cluster-resources/tree/main/helm-chart). Run the following commands to create the required resources in your cluster. You can find your agent access token on the **Workspace → Settings → Cluster → New Cluster** page. VESSL's cluster agent will automatically add your Kubernetes cluster.&#x20;

```
helm repo add vessl https://vessl-ai.github.io/cluster-resources/helm-chart
helm repo update
helm install vessl vessl/cluster-resources \
  --set namespace=vessl \
  --create-namespace \
  --set agent.accessToken=YOUR_AGENT_ACCESS_TOKEN
```

For those who are registering an on-premise cluster and want to [prevent unprivileged access to GPUs in Kubernetes](https://docs.google.com/document/d/1zy0key-EL6JH50MZgwg96RPYxxXXnVUdxLZwGiyqLd8/edit), we added `volumeListStrategy` as Helm values. You can set the values using the following script.&#x20;

```
helm install vessl vessl/cluster-resources \
  --set namespace=vessl \
  --create-namespace \
  --set agent.accessToken=YOUR_AGENT_ACCESS_TOKEN \
  --set nvidiaDevicePlugin.deviceListStrategy=volume-mounts
```

{% hint style="info" %}
You can also find this instruction on VESSL's Web Console.
{% endhint %}
