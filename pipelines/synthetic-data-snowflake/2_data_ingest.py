import argparse
import os

import pandas as pd

from cryptography.hazmat.primitives.serialization import load_pem_private_key
from snowflake.connector import connect
from snowflake.core import Root
from snowflake.core.table import Table, TableColumn
from snowflake.snowpark import Session


account = os.environ["SNOWFLAKE_ACCOUNT"]
user = os.environ["SNOWFLAKE_USER"]
private_key = load_pem_private_key(os.environ["SNOWFLAKE_PRIVATE_KEY"].replace("\\n", "\n").encode(), password=None)
warehouse = os.environ["SNOWFLAKE_WAREHOUSE"]

connection_parameters = {
    "account": account,
    "user": user,
    "private_key": private_key,
    "warehouse": warehouse
}


def create_table(database: str, schema: str, table_name: str, vector_dim: int):
    conn = connect(**connection_parameters)
    root = Root(conn)

    docs_table = Table(
        name=table_name,
        columns=[
            TableColumn(name="question", datatype="varchar"),
            TableColumn(name="answer", datatype="varchar"),
            TableColumn(name="code_example", datatype="varchar"),
            TableColumn(name="reference", datatype="varchar"),
            TableColumn(name="vector", datatype=f"vector(float, {vector_dim})")
        ]
    )

    root.databases[database.upper()].schemas[schema.upper()].tables.create(docs_table, mode="or_replace")


def ingest_data(session: Session, data_path: str, database: str, schema: str, table_name: str):
    df = pd.read_csv(data_path)
    df.columns = [c.upper() for c in df.columns]
    print(df.describe())

    session.write_pandas(
        df=df,
        table_name=table_name.upper(),
        database=database.upper(),
        schema=schema.upper(),
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest FAQ samples into Snowflake.")
    parser.add_argument("--data-path", type=str, required=True, help="Data path to load")
    parser.add_argument("--database", type=str, required=True, default="VESSL_SUPPORT", help="Database name to ingest data")
    parser.add_argument("--schema", type=str, default="public", help="Schema name to ingest data")
    parser.add_argument("--table-name", type=str, default="faqs_public_web", help="Table name to create")
    parser.add_argument("--vector-dim", type=int, default=1024, choices=[768, 1024], help="Dimension of the embedding vector")
    parser.add_argument("--embedding-model", type=str, default="nv-embed-qa-4", help="Name of the embedding to use")
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist.")
    if not data_path.endswith(".csv"):
        raise ValueError(f"Data path {data_path} is not a CSV file.")

    print(f"** Creating table {args.database}.{args.schema}.{args.table_name} with vector dimension {args.vector_dim} **")
    create_table(
        database=args.database,
        schema=args.schema,
        table_name=args.table_name,
        vector_dim=args.vector_dim,
    )

    with Session.builder.configs(connection_parameters).create() as session:
        print(f"** Ingesting data from {args.data_path} into {args.database}.{args.schema}.{args.table_name} **")
        ingest_data(
            session=session,
            data_path=args.data_path,
            database=args.database,
            schema=args.schema,
            table_name=args.table_name,
        )

        print(f"** Generating embeddings for {args.database}.{args.schema}.{args.table_name} **")
        query = f"""
    UPDATE {args.database}.{args.schema}.{args.table_name}
    SET vector = SNOWFLAKE.CORTEX.EMBED_TEXT_{args.vector_dim}('{args.embedding_model}', question);
    """
        session.sql(query).collect()

    print("** Data ingestion complete **")

if __name__ == "__main__":
    main()
