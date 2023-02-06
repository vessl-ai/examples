import io
import urllib
import urllib.request
import pandas as pd

from PIL import Image
from io import BytesIO
from tqdm import tqdm
from bs4 import BeautifulSoup

if __name__ == "__main__":

    ## preprocessing data ->
    df = pd.read_csv('dummy/boared_ape.csv')
    df = df[["tokenId", "imageUrl", "url"]]
    df["prompt"] = " "
    df["path"] = ""
    num_data = 4000
    for i in tqdm(range(0, num_data)):

        link = df.iloc[i].imageUrl
        urllib.request.urlretrieve(link, "input/Bored_Ape_Yacht_Club/{}.png".format(df.iloc[i].tokenId))
        df.iloc[i, -1] = "input/Bored_Ape_Yacht_Club/{}.png".format(df.iloc[i].tokenId)
        # img = Image.open("input/Bored_Ape_Yacht_Club/{}.png".format(df.iloc[i].tokenId))
        #
        # img.show()
        link = df.iloc[i].url
        headers = {'User-Agent': 'Chrome/66.0.3359.181'}
        req = urllib.request.Request(link, headers=headers)
        f = urllib.request.urlopen(req)
        data = f.read()

        soup = BeautifulSoup(data, 'html.parser')

        prompt = "An ape with"
        for j in range(2, 8):

            try:
                property_type = soup.select_one(
                    "#Body\ assets-item-properties > div > div > div > div:nth-child({}) > a > div > div.Property--type".format(
                        j)).text
                property_item = soup.select_one(
                    "#Body\ assets-item-properties > div > div > div > div:nth-child({}) > a > div > div.Property--value".format(
                        j)).text
                if property_type == 'Hat' or property_type in property_item:
                    prompt = prompt + " " + property_item.lower() + " " + " and"
                else:
                    prompt = prompt + " " + property_item.lower() + " " + property_type.lower() + " and"
            except:
                continue
        if prompt[-3:] == "and":
            prompt = prompt[:-4]

        # print(prompt)
        df.iloc[i, -2] = prompt

    df.to_csv("ape_with_prompt_all.csv")

    ###

    df = pd.read_csv("ape_with_prompt_all.csv")

    df['image'] = ""
    df.rename(columns={'prompt': 'text'}, inplace=True)

    images = []
    for i in tqdm(range(0, num_data)):
        im = Image.open(df.iloc[i, -2])
        buf = BytesIO()
        im.save(buf, format=im.format)
        byte_im = buf.getvalue()

        images.append(dict({"bytes": byte_im, "path": None}))

    df.iloc[:num_data, -1] = images

    df_new = df.iloc[:num_data, :]

    df_new = df_new[["image", "text"]]
    df_new.to_parquet(f"bored_ape_nft_{num_data}.parquet")

    # df = pd.read_parquet("pokemon.parquet")
    #
    # print(type(df.iloc[0,0]))
    #
    # print(df)
    #
    # df = pd.read_parquet("data/bored_ape_nft.parquet")
    #
    # print(type(df.iloc[0,0]))
    #
    # print(df)
