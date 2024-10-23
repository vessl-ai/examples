# Generate Synthetic Data with Snowflake Cortex

This repository contains code and VESSL Pipeline manifest to generate, curate, and utilize synthetic data, leveraging [**Snowflake Cortex**](https://www.snowflake.com/en/data-cloud/cortex/), a suite of AI features that use large language models (LLMs) to understand unstructured data, answer freeform questions, and provide intelligent assistance.

Snowflake Cortex gives you instant access to industry-leading large language models (LLMs) trained by researchers at companies like Mistral, Reka, Meta, and Google, including Snowflake Arctic, an open enterprise-grade model developed by Snowflake. Since these LLMs are fully hosted and managed by Snowflake, using them requires no setup. Your data stays within Snowflake, giving you the performance, scalability, and governance you expect.

## Prerequisites: Snowflake Settings
You need a Snowflake account to execute this pipeline:

1. Create a Snowflake account if you don't have one.
2. [Configure key pair authentication](https://docs.snowflake.com/en/user-guide/key-pair-auth#configuring-key-pair-authentication) in your Snowflake account.
3. [Create a new secret](https://docs.vessl.ai/guides/organization/secrets), with name `snowflake-private-key`, in your VESSL organization. Its value is the private key of the key pair you created above.
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