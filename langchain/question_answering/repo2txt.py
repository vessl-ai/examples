# @jinpil
# github repository to text file

import glob
import os
import shutil
import subprocess
import sys
import argparse
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
import json

GITHUB_REPO_URL = 'https://github.com/vessl-ai/gitbook-docs'
REPO_DIR = './gitbook-docs'


def clone(repo_url):
    if repo_url is None:
        raise Exception("repo_url is None")
        exit(-1)
    else:
        subprocess.run(['git', 'clone', repo_url], check=True)
        return f"./{repo_url.split('/')[-1]}"

def get_docs(repo_dir, repo_url,):

    docs = [ ]
    assert (repo_dir or repo_url), "repo_dir or repo_url must be not None"

    if repo_url is not None:
        clone(repo_url)
        repo_dir = f"./{repo_url.split('/')[-1]}"

    # recursively get file -> load text -> docs list
    for file in glob.glob(f"{repo_dir}/**/*", recursive=True):
        if not os.path.isdir(file)  :
            try :
                loader = TextLoader(file, encoding=None )
                docs.extend(loader.load_and_split())
            except:
                pass
    return docs
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_url', type=str, default=None)
    parser.add_argument('--repo_dir', type=str, default=
    './gitbook-docs')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--file_type', type=str, nargs="+", default=['md', 'txt'])
    parser.add_argument('--mode', type=str, default='git')
    args = parser.parse_args()
    return args


def parse_slack_messages(file_path, output_dir) :
    """
    file_path : slack message file path
    add header about its directory info to eech file and save to output_dir

    sort by date and merge by channel is needed
    channels.json , integration_logs.json, users.json are not needed
    random, support, support-kr, vessl-tips, yonsei-ai, kaist-ai, general, random, etc

    json architecture
    data :{
        0 {
            type (message, file, etc)
            subtype (channel_join, etc) -> channel in의 경우에는 버림
            text (maybe the question) -> most important part
            blocks
            user_profile : {
                name
                }
        }

        1 {
            type :
            subtype :
            text (maybe answer) -> most important part
    """

    channels = set()
    channel_conversation_log = []
    for file in sorted(glob.glob(f"{file_path}/**/*.json", recursive=True)):

        if (os.path.isdir(file)) :
            continue;

        channel = file.split("/")[-2]
        channels.add(channel)
        target_file = os.path.join(output_dir , channel + '.txt')
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        datas = json.load(open(file, 'r', encoding='utf-8'))
        for data in datas :
            if 'text' in data and data['text'] != '' and  (not 'subtype' in data  or data["subtype"] != 'channel_join') :
                with open(target_file, 'a', encoding='utf-8') as f:
                    f.write('-----------------------------------------' + '\n')
                    if ("user_profile" in data and data['user_profile']['name'] != '') :
                        f.write(data['user_profile']['name'] + '\n')
                    else :
                        f.write("anonymous\n")
                    f.write(data['text'] + '\n')

if __name__ == "__main__":

    args = parse_args()
    # convert(parse_args())
    parse_slack_messages("./vessl_community_slack_2023_5_4", "merge_data/slack_2023_5_4")
