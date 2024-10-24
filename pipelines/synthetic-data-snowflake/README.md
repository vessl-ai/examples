# Generate Synthetic Data with Snowflake Cortex

This repository contains code and VESSL Pipeline manifest to generate, curate, and utilize synthetic data, leveraging [**Snowflake Cortex**](https://www.snowflake.com/en/data-cloud/cortex/), a suite of AI features that use large language models (LLMs) to understand unstructured data, answer freeform questions, and provide intelligent assistance.

Snowflake Cortex gives you instant access to industry-leading large language models (LLMs) trained by researchers at companies like Mistral, Reka, Meta, and Google, including Snowflake Arctic, an open enterprise-grade model developed by Snowflake. Since these LLMs are fully hosted and managed by Snowflake, using them requires no setup. Your data stays within Snowflake, giving you the performance, scalability, and governance you expect.

## Prerequisites: Snowflake Settings
> **Note**: This example uses Snowflake's computing resource and storage, and consumes credits. 

You need a Snowflake account to execute this pipeline:

1. Create a Snowflake account if you don't have one.
2. [Configure key pair authentication](https://docs.snowflake.com/en/user-guide/key-pair-auth#configuring-key-pair-authentication) in your Snowflake account.
3. [Create a new secret](https://docs.vessl.ai/guides/organization/secrets) in your VESSL organization with name `snowflake-private-key`. Its value is the private key of the key pair you created above.
    > **Note:** Since secrets only accepts single line texts, you have to replace new lines in the private key with a newline character(`\n`).
4. Create a database and a schema in your Snowflake account, to save the synthetic data.
5. Create a warehouse if you don't have one.

## Running Locally
1. Clone this repository and install dependencies:
    ```sh
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/pipelines/synthetic-data-snowflake
    $ pip install -r requirements.txt
    $ pip install gradio
    ```

2. Export Snowflake credentials as environment variables:
    ```sh
    $ export SNOWFLAKE_ACCOUNT={Snowflake account identifier}
    $ export SNOWFLAKE_USER={Snowflake account user name}
    $ export SNOWFLAKE_PRIVATE_KEY={Snowflake private key}
    $ export SNOWFLAKE_WAREHOUSE={Snowflake warehouse name}
    ```

3. Run the python files one by one:
    ```sh
    $ python 1_generate_synthetic_data.py
    $ python 2_data_curation.py --input-path samples.csv --output-path samples-curated.csv
    $ python 3_data_ingest.py \
        --data-path samples-curated.csv \
        --database {database name} \
        --schema {schema name}
    $ python 4_chat.py --database {database name} --schema {schema name}
    ```

## Running with VESSL Pipeline
VESSL Pipeline is a tool designed for those working on streamlining complex machine learning workflows. By focusing on automation, it reduces manual intervention, making it especially useful for workflows that require consistent and repeated executions.

To create and run the synthetic data generation pipeline, follow the steps below:

1. Create a VESSL acccount if you don't have one.
2. Go to the VESSL Pipelines page. Create a new pipeline.
    ![Create New Pipeline](./assets/new-pipeline.png)
3. Create a new revision at the **Revisions** tab.
    ![Create New Revision](./assets/new-revision.png)
4. Enable the **Create from YAML** toggle and upload the YAML file in this folder ([`pipeline.yaml`](./pipeline.yaml)).
5. Click **Edit** button of the revision you just created.
    ![Edit revision](./assets/revision-edit.png)
6. Click **Publish** button to publish the revision.
    ![Publish revision](./assets/revision-publish.png)
7. Click **Run** button of the revision you just published.
    ![Run revision](./assets/revision-run.png)
8. Enter input variables.