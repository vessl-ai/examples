import os
import base64

import streamlit as st
from PIL import Image
from io import BytesIO

from model import get_antelope_app, load_model, run_pipe

import boto3
from botocore.exceptions import ClientError

def send_email(html):
    client = boto3.client('ses',region_name='us-west-2')
    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    'vesslai@icloud.com',
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': 'UTF-8',
                        'Data': html,
                    }
                },
                'Subject': {
                    'Charset': 'UTF-8',
                    'Data': 'Photo request',
                },
            },
            Source='seokju@vssl.ai',
        )
    # Display an error if something goes wrong.
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])

st.title("Convert your photo to sticker!😃")

pipe = load_model()
app = get_antelope_app()

TEMP_PATH = "temp/image.png"


def crop_and_resize(image: Image.Image, target_length: int = 1024):
    w = image.width
    h = image.height

    # crop
    if w > h:
        side = h
        left = w // 2 - side // 2
        cropped = image.crop((left, 0, left + side, side))
    else:
        side = w
        top = h // 2 - side // 2
        cropped = image.crop((0, top, side, top + side))

    # resize
    new_image = cropped.resize((target_length, target_length), Image.Resampling.LANCZOS)

    return new_image


with st.sidebar:
    prompt = st.text_input(
        "prompt", value="Sticker, cel shaded, svg, vector art, sharp"
    )
    negative_prompt = st.text_input(
        "negative prompt",
        value="photo, photography, soft, nsfw, nude, ugly, broken, watermark, oversaturated",
    )
    # strength = st.slider(
    #     "denoising strength", min_value=0.0, max_value=1.0, step=0.01, value=0.4
    # )
    guidance = st.slider("cfg", min_value=1.0, max_value=20.0, step=0.1, value=7.5)
    steps = st.number_input("inference steps", 10, 200, step=1, value=20)
    image_size = st.number_input(
        "image size", min_value=512, max_value=768, step=64, value=768
    )
    num_photos = st.number_input("number of photos", 1, 4, step=1, value=4)


st.header("Upload image or take your photo!")
with st.form("image", clear_on_submit=False):
    image_path = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
    img_file_buffer = st.camera_input("Take your photo!")
    submit_button = st.form_submit_button(label="Generate")

    os.makedirs("./temp", exist_ok=True)

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        img.save(TEMP_PATH, "PNG")
    elif image_path is not None:
        img = Image.open(image_path)
        img.save(TEMP_PATH, "PNG")

if submit_button:
    st.header("Result Image")

    input_image = Image.open(TEMP_PATH)
    # input_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    width, height = input_image.size

    input_image = crop_and_resize(input_image, image_size)

    with st.spinner("generating images..."):
        images = run_pipe(
            pipe,
            app,
            input_image,
            prompt,
            negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_photos,
        )

    stream = BytesIO()
    images.save(stream, format="png")
    stream.seek(0)
    imgObj=stream.read()
    b64img=base64.b64encode(imgObj)
    mail=f'''<html>
<body> 
<h1>Here is your sticker!</h1>
<img src="data:image/png;base64,{str(b64img)}" alt="sticker">
</body>
</html>'''
    send_email(mail)


    st.image(images)

    # send_email_button = st.button("Send email")
