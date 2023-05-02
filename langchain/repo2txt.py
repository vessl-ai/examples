# @jinpil
# github repository to text file

import glob
import os
import shutil
import subprocess
import sys
import argparse

GITHUB_REPO_URL = 'https://github.com/vessl-ai/gitbook-docs'
REPO_DIR = './gitbook-docs'


def clone(repo_url):
    if repo_url is None:
        raise Exception("repo_url is None")
        exit(-1)
    else:
        subprocess.run(['git', 'clone', repo_url], check=True)
        return f"./{repo_url.split('/')[-1]}"


def convert(args):
    repo_dir = args.repo_dir
    repo_url = args.repo_url
    output_dir = args.output_dir
    file_type = args.file_type

    if not os.path.exists(output_dir) :
        os.mkdir(output_dir)

    assert (repo_dir or repo_url), "repo_dir or repo_url must be not None"

    if repo_dir is None and repo_url is not None:
        clone(repo_url)
        repo_dir = f"./{repo_url.split('/')[-1]}"

    ## recursively get file(in this case .py and .md list)
    for ftype in file_type:

        print(f"Converting {ftype} files in {repo_dir} to {output_dir}, "
              f"{len(glob.glob(f'{repo_dir}/**/*.{ftype}', recursive=True))} files found")
        print("--------------------------------------------------")

        for file in glob.glob(f"{repo_dir}/**/*.{ftype}", recursive=True):
            target_file = str(output_dir + file[1:].replace(ftype, 'txt'))
            print(f"Converting {file} to {target_file}")
            # make directory
            os.makedirs(os.path.dirname(target_file), exist_ok=True)

            # write files to target_file
            print(file, target_file)
            with open(file, 'r', encoding='utf-8') as f:
                md_text = f.read()
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(md_text)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_url', type=str, default='https://github.com/vessl-ai/gitbook-docs')
    parser.add_argument('--repo_dir', type=str, default=
    './gitbook-docs')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--file_type', type=str, nargs="+", default=['md', 'txt'])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    convert(parse_args())
