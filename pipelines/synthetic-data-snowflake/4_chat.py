import argparse
import os
from time import sleep
from typing import List

from cryptography.hazmat.primitives.serialization import load_pem_private_key
import gradio as gr
from snowflake.snowpark import Session
from snowflake.core import Root
from snowflake.core.table import Table, TableColumn

account = os.environ["SNOWFLAKE_ACCOUNT"]
user = os.environ["SNOWFLAKE_USER"]
private_key = load_pem_private_key(
    os.environ["SNOWFLAKE_PRIVATE_KEY"].replace("\\n", "\n").encode(), password=None
)
warehouse = os.environ["SNOWFLAKE_WAREHOUSE"]


class LLMChatHandler:
    def __init__(
        self,
        database: str,
        schema: str,
        table: str,
        vector_dim: int,
        embedding_model: str,
        llm_model: str,
    ) -> None:
        self.database = database
        self.schema = schema
        self.table = table
        self.vector_dim = vector_dim
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.connection_parameters = {
            "account": account,
            "user": user,
            "private_key": private_key,
            "warehouse": warehouse,
            "database": self.database,
            "schema": self.schema,
        }
        self.session = None

    def rag_query(self, message: str, history: List[List[str]]) -> str:

        self.session = Session.builder.configs(self.connection_parameters).create()
        root = Root(self.session)
        docs_table = Table(
            name="query_table",
            columns=[
                TableColumn(
                    name="prompt_vector", datatype=f"vector(float, {self.vector_dim})"
                )
            ],
        )
        root.databases[self.database].schemas[self.schema].tables.create(
            docs_table, mode="or_replace"
        )

        self.session.sql(
            f"INSERT INTO query_table SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_{self.vector_dim}('{self.embedding_model}', '{message}')"
        ).collect()

        result = self.session.sql(
            f"""
    WITH
        result AS (
            SELECT
                d.answer,
                VECTOR_COSINE_SIMILARITY(d.vector, q.prompt_vector) AS similarity
            FROM {self.table} d, query_table q
            ORDER BY similarity DESC
            LIMIT 5
        ),

        result_agg AS (
            SELECT
                LISTAGG(answer, '\n') AS context,
                '{message}' AS prompt
            FROM result
        )

    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        '{self.llm_model}',
        [
            {{
                'role': 'system',
                'content': 'You are a support engineer in VESSL AI. Your task is to answer the question from the client and help them better understand VESSL AI and its solutions(VESSL Run, VESSL Services and VESSL Pipeline) better. Answer the <question> based on the <context> given. If there is no information in the <context>, specify that you cannot answer the question.'
            }},
            {{
                'role': 'system',
                'content': CONCAT('<context>', context, '</context>')
            }},
            {{
                'role': 'user',
                'content': CONCAT('<question>', prompt, '</question>')
            }}
        ],
        {{
            'temperature': 0.2
        }}
    ) FROM result_agg;
        """
        ).to_pandas()
        self.session.close()
        return eval(result.iloc[0, 0])["choices"][0]["messages"]

    def close_app(self):
        gr.Info("Terminated the app!")
        sleep(1)
        os._exit(0)


def main(args):
    handler = LLMChatHandler(
        args.database,
        args.schema,
        args.table,
        args.vector_dim,
        args.embedding_model,
        args.llm_model,
    )

    with gr.Blocks(
        title=f"❄️ Cortex Chatbot with {args.llm_model}", fill_height=True
    ) as demo:
        with gr.Row():
            gr.Markdown(
                f"<h2>Chatbot with ❄️ Cortex and {args.llm_model}</h2>"
                "<h3>Interact with LLM using chat interface!<br></h3>"
            )
        gr.ChatInterface(handler.rag_query)
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(
                fn=lambda: gr.update(interactive=False), outputs=[close_button]
            ).then(fn=handler.close_app)

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat with LLM using Snowflake Cortex",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="VESSL_SUPPORT",
        help="Database name where the data is",
    )
    parser.add_argument(
        "--schema", type=str, default="public", help="Schema name where the data is"
    )
    parser.add_argument(
        "--table",
        type=str,
        default="faqs_public_web",
        help="Table name where the data is",
    )
    parser.add_argument(
        "--vector-dim",
        type=int,
        default=1024,
        choices=[768, 1024],
        help="Dimension of the embedding vector",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nv-embed-qa-4",
        help="Name of the embedding to use",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.1-405b",
        help="name of the LLM model to use",
    )
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    args = parser.parse_args()

    main(args)
