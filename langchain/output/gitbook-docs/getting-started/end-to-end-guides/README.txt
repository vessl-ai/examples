# End-to-end Guides

### Overview

VESSL AI provides end-to-end machine learning services for dataset mount, machine learning experiment training, hyperparameter sweep, and model managing. This document will guide you to the general end-to-end workflow of how to use those features using VESSL CLI and SDK thoroughly.&#x20;

### Prerequisites

In the tutorials below, an organization and a project on VESSL are required. We recommend creating an organization and a project in VESSL Web Console. For convenience, the organization name will be `YOUR_ORGANIZATION_NAME` and the project name will be `YOUR_PROJECT_NAME`. Just replace them with the organization name and project name you created, respectively.

#### 1. Create an organization

Once you have created a VESSL account, the first thing you need to do is create an organization. Refer to the [create an organization](../../user-guide/organization/create-an-organization.md) page in the VESSL Web Console to create an organization.

#### 2. Create a project

After the organization setup is complete, you need to create a machine learning project. Similarly, refer to the [create a project](../../user-guide/project/creating-a-project.md) page in the Web Console and create a project with a cool project name.

#### 3. Install VESSL Client

To use VESSL CLI or SDK, you need to install VESSL Client. Execute the following command in your terminal.

```bash
pip install vessl
```

For more information on installing the VESSL Client, see [What is the VESSL CLI/SDK?](../../api-reference/what-is-the-vessl-cli-sdk.md).

### Workflows

After successfully creating the organization and project in VESSL Web Console and installing VESSL Client, let's start the machine learning journey with VESSL.&#x20;

{% content-ref url="cli-driven-workflow.md" %}
[cli-driven-workflow.md](cli-driven-workflow.md)
{% endcontent-ref %}

{% content-ref url="sdk-driven-workflow.md" %}
[sdk-driven-workflow.md](sdk-driven-workflow.md)
{% endcontent-ref %}
