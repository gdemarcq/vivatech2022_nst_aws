import streamlit as st
import numpy as np
import base64
import requests
import json
import io
from PIL import Image
from io import BytesIO
import os

# Due to streamlit design, we must use SessionState to save
# some global variables in order to avoid reseting

if "update" not in st.session_state:
    st.session_state.update = True

if "result_img" not in st.session_state:
    st.session_state.result_img = None

def on_input_change():
    st.session_state.update = True


def on_download_clicked():
    print("download")
    st.session_state.update = False


def pil_image_to_b64str(im):
    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    img_bytes = base64.b64encode(buffered.getvalue())
    img_str = img_bytes.decode('utf-8')
    return img_str


def b64str_to_numpy(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytes_obj = io.BytesIO(base64bytes)
    img = Image.open(bytes_obj)
    return np.asarray(img)


def bytesio_obj_to_b64str(bytes_obj):
    return base64.b64encode(bytes_obj.read()).decode("utf-8")


def bytesio_to_pil_image(bytes_obj):
    data = io.BytesIO(bytes_obj.read())
    img = Image.open(data)
    return img


def numpy_to_pil_image(np_img):
    if np_img.dtype in ["float32", "float64"]:
        return Image.fromarray((np_img * 255 / np.max(np_img)).astype('uint8'))
    else:
        return Image.fromarray(np_img)


def make_payload(img_base64, method, model):
    headers = {"Content-Type": "application/json"}

    body = {
        "inputs": [
            {"type": "IMAGE", "data": img_base64},
        ],
        "outputs": [
            {"task_name": "ocv_convert_to", "task_index": 0, "output_index": 0},
        ],
        "parameters": [
            {"task_name": "infer_neural_style_transfer", "task_index": 0, "parameters":
                {"method": method, "model": model}
             },
        ],
        "isBase64Encoded": True
    }

    payload = json.dumps(body)
    return headers, payload


def run_workflow_on_image(url, image, method, model):
    img_base64 = pil_image_to_b64str(image)
    headers, payload = make_payload(img_base64, method, model)
    response = requests.put(url, headers=headers, data=payload)
    data_dict = response.json()
    img_out = None

    if "body" in data_dict:
        data_dict = json.loads(data_dict["body"])

    if "outputs" in data_dict:
        if len(data_dict["outputs"]) == 0:
            raise RuntimeError(data_dict["message"])
        else:
            for out in data_dict["outputs"]:
                if out["type"] == "IMAGE":
                    img_out = b64str_to_numpy(out["data"])

    return img_out

def demo():
    st.set_page_config(layout="wide")

    # ------------ Sidebar
    st.sidebar.image("./images/vivatech.png", use_column_width=True)
    st.sidebar.image("./images/ikomia_logo_512x512.png", use_column_width=True)
    input_img = None
    uploaded_input = st.sidebar.file_uploader("Choose input image", on_change=on_input_change)
    if uploaded_input is not None:
        input_img = bytesio_to_pil_image(uploaded_input)
    # ------------ Main page TITLE
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("./images/banner.png")

    st.markdown("<h1 style='text-align: center; color: white;'>Free image stylization of your photos !</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: grey;'>Choose your favorite painting and download your creation</h2>", unsafe_allow_html=True)

    # ------------ Main page BODY
    col1, col2, col3 = st.columns([1, 3, 3])

    # Create streamlit placeholder
    img_widget_1 = col1.empty()
    img_widget_2 = col2.empty()
    img_widget_3 = col3.empty()
    download_bt = col3.empty()

    # Grab the paths to all neural style transfer models in our 'models'
    # directory, provided all models end with the '.t7' file extension
    models_eccv16 = ["starry_night", "la_muse", "the_wave", "composition_vii"]
    models_instance_norm = ["udnie", "mosaic", "candy", "the_scream", "feathers"]
    models = models_eccv16 + models_instance_norm

    combo_model = []
    for i in range(len(models)):
        tmp = os.path.basename(models[i])
        _name = os.path.splitext(tmp)[0]
        combo_model.append(_name)

    model = col1.selectbox('Which painting ?', combo_model)
    img_path = "./painting/"+model+".jpg"
    img_style = Image.open(img_path)

    # Update image in streamlit view
    img_widget_1.image(img_style,
                       use_column_width=True,
                       clamp=True
                       )

    if model in models_eccv16:
        method = "eccv16"
    else:
        method = "instance_norm"

    if input_img is not None:
        img_widget_2.image(input_img,
                           use_column_width=True,
                           clamp=True
                           )

        if st.session_state.update:
            print("process")
            url = "https://psjzth8q2l.execute-api.eu-west-1.amazonaws.com/"
            img_out = run_workflow_on_image(url, input_img, method, model)

            if img_out is not None:
                # Replace the current frame by the modified one
                img_pil = numpy_to_pil_image(img_out)
                data = BytesIO()
                img_pil.save(data, format="PNG")

                download_bt.download_button(
                    label="Download image",
                    data=data,
                    file_name="your_image.png",
                    mime="image/png",
                    on_click=on_download_clicked
                )
                result = img_out
            else:
                result = input_img

            img_widget_3.image(result,
                               use_column_width=True,
                               clamp=True
                               )
        else:
            img_widget_3.image(st.session_state.result_img,
                               use_column_width=True,
                               clamp=True
                               )

if __name__ == '__main__':
    demo()