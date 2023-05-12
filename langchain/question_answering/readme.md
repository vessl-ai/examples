# Making Question Answering Chatbot On Any Clouds(AWS, GCP, On-premise) with VESSL

## [1] Start with VESSL 
### You can make server that can access from anywhere

### 0. Get your OpenAI API, Prepare your data (eg. books for study, documents on github, etc)
###### Get your own OpenAI API key from [here](https://platform.openai.com/account/api-keys)


### 1. Sign up to [VESSL](https://vessl.ai)
![plot](../imgs/signup.png)

### 2. Make workspace (only cpu is enough)
###### You can connect your own GPU server or AWS, GCP cluster to VESSL

##### 2.1  Expose port 8501 for streamlit in advanced setting
![plot](../imgs/workspace.png)
![plot](../imgs/workspace_make.png)
##### 2.2 Click jupyter notebook logo after workspace status changed to running
![plot](../imgs/click_jupyter_notebook.png)

### 3. Run the code below in jupyter notebook cli terminal
```bash
git clone  "https://github.com/vessl-ai/examples.git" && cd /root/examples/langchain/question_answering/ && bash ./run.sh
```
![plot](../imgs/jupyter_notebook.png)

### 4. Do not Click Link that streamlit gives, just modify the port of jupyternotebook link from 8888 to 8501 and click it

![plot](../imgs/jupyter_link.png)
![plot](../imgs/streamlit_link.png)

### 5. You can see the question answering chatbot, follow its instruction
![plot](../imgs/streamlit_demo.png)

## [2] Start with your local device

```bash
git clone  "https://github.com/vessl-ai/examples.git"
cd examples/langchain/question_answering
./run.sh
```

