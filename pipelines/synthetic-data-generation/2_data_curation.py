import argparse
import os
from typing import Union

import gradio as gr
import pandas as pd


class Curator:
    def __init__(self, input_path: str, output_path: str, rows_per_page: int) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.rows_per_page = rows_per_page
        self.df = None
        self.total_pages = None
        self.current_page = None
        self.selected_qas = []

        self._load_data()

    def _load_data(self) -> None:
        df = pd.read_csv(self.input_path)
        df["NO"] = df.index + 1
        self.df = df[["NO", "question", "answer", "code_example", "reference"]]
        self.total_pages = len(df) // self.rows_per_page + 1
        self.current_page = 0

    def select(self, dataframe: gr.DataFrame, evt: gr.SelectData) -> str:
        qa_num = dataframe.iloc[evt.index[0], 0]
        qa_num = str(qa_num)
        if qa_num in self.selected_qas:
            self.selected_qas.remove(qa_num)
        else:
            self.selected_qas.append(qa_num)

        return "\n".join(self.selected_qas)

    def move_page(self, page_count: int) -> Union[gr.Button, gr.DataFrame]:
        self.current_page += page_count
        if self.current_page < 0:
            self.current_page = 0
        elif self.current_page >= self.total_pages:
            self.current_page = self.total_pages - 1

        return (
            gr.update(value=f"Page {self.current_page + 1} / {self.total_pages}"),
            gr.DataFrame(
                self.df[
                    self.rows_per_page * self.current_page : self.rows_per_page * (self.current_page + 1)
                ]
            ),
        )

    def save(self):
        if self.selected_qas:
            index_to_delete = [int(i) - 1 for i in self.selected_qas]
            df = self.df.drop(index_to_delete, axis=0)
        df[["question", "answer", "code_example", "reference"]].to_csv(
            self.output_path, index=False
        )
        return f"Saved {len(df)} rows to {self.output_path}."


def main(args):
    curator = Curator(args.input_path, args.output_path, args.rows_per_page)
    with gr.Blocks(title=f"Curation User Review ({args.input_path})") as demo:
        with gr.Row():
            gr.Markdown("<h2>Selected rows will be deleted.</h2>")
        page_num_btn = gr.Button(
            f"Page {curator.current_page + 1} / {curator.total_pages}",
            interactive=False,
            variant="secondary",
        )
        grdf = gr.DataFrame(
            curator.df[
                args.rows_per_page * curator.current_page : args.rows_per_page * (curator.current_page + 1)
            ],
            max_height=1000,
            interactive=False,
        )
        with gr.Row():
            prev_page_btn = gr.Button("previous page")
            next_page_btn = gr.Button("next page")
        selected_textbox = gr.Textbox(
            interactive=False, label="numbers of the QA pairs to delete"
        )
        with gr.Row():
            proceed_btn = gr.Button(f"Proceed with selected rows")
            confirm_proceed_btn = gr.Button(f"Confirm selected rows", visible=False)
            cancel_proceed_btn = gr.Button("Cancel", visible=False)
            abort_btn = gr.Button("Abort")
            confirm_abort_btn = gr.Button(
                "Abort the synthetic data creation workflow", visible=False
            )
            cancel_abort_btn = gr.Button("Cancel", visible=False)

        grdf.select(curator.select, grdf, selected_textbox)

        prev_page_btn.click(lambda: curator.move_page(-1), outputs=[page_num_btn, grdf])
        next_page_btn.click(lambda: curator.move_page(1), outputs=[page_num_btn, grdf])

        # Bind the events to the proceed, confirm and cancel button
        # Confirm button saves the selected images to the output path
        proceed_btn.click(
            lambda: [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            ],
            None,
            [proceed_btn, abort_btn, confirm_proceed_btn, cancel_proceed_btn],
        )
        cancel_proceed_btn.click(
            lambda: [
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ],
            None,
            [proceed_btn, abort_btn, confirm_proceed_btn, cancel_proceed_btn],
        )
        confirm_proceed_btn.click(
            lambda: [
                gr.update(interactive=False, visible=True),
                gr.update(interactive=False, visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ],
            None,
            [proceed_btn, abort_btn, confirm_proceed_btn, cancel_proceed_btn],
        ).then(fn=curator.save, inputs=None, outputs=[proceed_btn]).then(
            fn=lambda: gr.Info("Deleted selected rows.")
        ).then(
            lambda: os._exit(0)
        )

        # Bind the events to the abort button
        abort_btn.click(
            lambda: [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            ],
            None,
            [proceed_btn, abort_btn, confirm_abort_btn, cancel_abort_btn],
        )
        cancel_abort_btn.click(
            lambda: [
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ],
            None,
            [proceed_btn, abort_btn, confirm_abort_btn, cancel_abort_btn],
        )
        confirm_abort_btn.click(
            lambda: [
                gr.update(interactive=False, visible=False),
                gr.update(interactive=False, visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ],
            None,
            [proceed_btn, abort_btn, confirm_abort_btn, cancel_abort_btn],
        ).then(lambda: gr.Info("Aborted synthetic data creation pipeline!")).then(
            lambda: os._exit(1)
        )

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic data curator",
    )
    parser.add_argument(
        "--input-path",
        required=True,
        type=str,
        help="input data path (has to end with .csv)",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=str,
        help="output data path (has to end with .csv)",
    )
    parser.add_argument(
        "--rows-per-page",
        default=10,
        type=int,
        help="how many rows to show in one page",
    )
    parser.add_argument(
        "--port", default=7860, type=int, help="port number for the Gradio app"
    )
    args = parser.parse_args()

    main(args)
