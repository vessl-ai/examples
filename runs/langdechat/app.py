import json
import os
import uuid
from typing import List

import gradio as gr
import requests

LANGDECHAT_ENDPOINT = "http://localhost:8000"

if os.environ.get("LANGDECHAT_ENDPOINT"):
    LANGDECHAT_ENDPOINT = os.environ.get("LANGDECHAT_ENDPOINT")

def format_history(history: List[str]):
    formatted = []
    print(">>>>>>>>>> History")
    print(history)
    for h in history:
        print(h)
        formatted.append({
            "role": "user",
            "content": h[0]
        })
        formatted.append({
            "role": "agent",
            "content": h[1]
        })
    return formatted

continuation_token: str = None

def query_langdechat(message: str, history: List[str], endpoint: str, continuation_token: str, api_key: str):
    data = {
        "query": message,
        # "history": format_history(history),
        "variables": {}
    }

    if continuation_token:
        data["continuation_token"] = continuation_token

    print(">>>>>>>>>> Data")
    print(json.dumps(data))
    headers = {}
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}

    r = requests.post(f"{endpoint}/chat", json=data, headers=headers)

    if r.status_code != 200:
        print(f">>>>>>>>>> Error: {r.status_code}")
        print(r.text)
        return "Error"
    res_data = r.json()
    continuation_token = res_data["continuation_token"]
    return res_data["message"]

def query_langdechat_test(message: str, history: List[str], endpoint: str, continuation_token: str, workflow_spec: str, user_group_id: int):
    spec_json = json.loads(workflow_spec)

    data = {
        "query": message,
        # "history": format_history(history),
        "variables": {},
        "workflow_spec": spec_json['workflow_revision_input'], #some workflow_revision_input
    }

    if continuation_token:
        data["continuation_token"] = continuation_token
    if user_group_id:
        data["user_group_id"] = int(user_group_id)

    print(">>>>>>>>>> Data")
    print(json.dumps(data))

    r = requests.post(f"{endpoint}/chat_test", json=data)

    if r.status_code != 200:
        print(f">>>>>>>>>> Error: {r.status_code}")
        print(r.text)
        return "Error"
    res_data = r.json()
    continuation_token = res_data["continuation_token"]
    return res_data["message"]

with gr.Blocks() as demo:
    endpoint = gr.Textbox(LANGDECHAT_ENDPOINT, label="Endpoint")
    with gr.Tab("deploy-mode"):
        api_key = gr.Textbox("api_key", label="API Key")
        gr.ChatInterface(
            fn=query_langdechat,
            examples=[],
            title="Langdechat Test Bot",
            additional_inputs=[endpoint, gr.State(value=lambda: uuid.uuid4().hex.lower()[0:10]), api_key],
        )
    with gr.Tab("test-mode"):
        workflow_revision_input = gr.TextArea("workflow_revision_input", label="workflow_revision_input")
        user_group_id = gr.Text("", label="user_group_id (optional)")
        gr.ChatInterface(
            fn=query_langdechat_test,
            examples=[],
            title="Langdechat Test Bot",
            additional_inputs=[endpoint, gr.State(value=lambda: uuid.uuid4().hex.lower()[0:10]), workflow_revision_input, user_group_id],
        )

port=4999
if os.environ.get("PORT"):
    port = int(os.environ.get("PORT"))

demo.launch(server_name="0.0.0.0", server_port=port)