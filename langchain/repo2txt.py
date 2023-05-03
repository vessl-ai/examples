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


if __name__ == "__main__":

    args = parse_args()

    get_docs(args.repo_dir, args.repo_url)
