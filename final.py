# Core Pkgs
import streamlit as st
st.set_page_config(page_title="Tool for Lung Classification from Chest X-Ray", page_icon="covid19.jpeg", layout='centered', initial_sidebar_state='auto')

import os
import time
from collections import namedtuple

import albumentations as A
import torch
from torch._prims_common import DeviceLikeType
from torch import nn
from torch.utils import model_zoo

import unet as Unet
# Viz Pkgs
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
# from model import model 

# AI Pkgs
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

# Loading the tflite model
model_path = "App/main.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']

# Define class names
class_names = ['Covid', 'Viral Pneumonia', 'Normal']

def main():
    """Simple Tool for Lung Classification from Chest X-Ray"""
    html_templ = """
    <div style="background-color:dark;padding:10px;">
    <h1 style="color:white">Covid-19 Classification Tool</h1>
    </div>
    """

    st.markdown(html_templ,unsafe_allow_html=True)
    st.write("A simple proposal for Covid-19 Diagnosis powered by Deep Learning and Streamlit")

    image_file = st.sidebar.file_uploader("Upload an X-Ray Image (jpg, png or jpeg)",type=['jpg','png','jpeg'])

    if image_file is not None:
        our_image = Image.open(image_file)

        if st.sidebar.button("Image Preview"):
            st.sidebar.image(our_image,width=300)

        activities = ["Image Enhancement","Diagnosis", "Lung Segment"]
        choice = st.sidebar.selectbox("Select Activty",activities)

        if choice == 'Image Enhancement':
            st.subheader("Image Enhancement")
            enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Contrast", "Brightness"])
            if enhance_type == "Contrast":
                c_rate = st.slider("Contrast", 0.5, 5.0)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output, width=600, use_column_width=True)
            elif enhance_type == "Brightness":
                c_rate = st.slider("Brightness", 0.5, 5.0)
                enhancer = ImageEnhance.Brightness(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output, width=600, use_column_width=True) 
            else:
                st.text("Original Image")
                st.image(our_image, width=600, use_column_width=True)  

        elif choice == 'Diagnosis':
            if st.sidebar.button("Diagnosis"):
                new_img = np.array(our_image.convert('RGB')) 
                new_img = cv2.cvtColor(new_img,1)
                gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                st.text("Chest X-Ray")
                st.image(gray, width=400, use_column_width=True)

                # Image reshaping according to tensorflow format
            img = our_image.resize((input_shape[1], input_shape[2]))
            img = np.array(img, dtype=np.float32)
            img /= 255.0
            img = np.expand_dims(img, axis=0)

                # Predict function
            def predict(image):
                image = image.convert("RGB").resize((input_shape[1], input_shape[2])) 
                # Image reshaping according to tensorflow format INSIDE predict function
                image = image.resize((input_shape[1], input_shape[2]))
                image = np.array(image)  # Convert to NumPy array
                if len(image.shape) == 2:  # Check if it's grayscale
                    image = np.stack((image,)*3, axis=-1)  # Convert to 3 channels
                image = np.expand_dims(image, axis=0)  # Add batch dimension
                image = image / 255.0  # Normalize pixel values

                interpreter.set_tensor(input_details[0]['index'], image.astype(input_dtype))
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                predicted_class_index = np.argmax(predictions, axis=1)
                predicted_class_name = class_names[predicted_class_index[0]]
                return predicted_class_name
            
            predicted_class_name = predict(our_image) 
                                                    
                
            st.markdown(f"Classified as: <span style='font-style: italic; font-weight: bold;'>{predicted_class_name}",unsafe_allow_html=True) 
            st.warning("This Web App is just a DEMO about Streamlit and Artificial Intelligence and there is no clinical value in its diagnosis!")
            
        else:
            st.subheader("Lung Segmentation")
            # st.set_option("deprecation.showfileUploaderEncoding", False)

            img_size = 512
            aug = A.Compose([A.Resize(img_size, img_size, interpolation=1, p=1)], p=1)

            model = namedtuple("model", ["url", "model"])

            models = {
                "resnet34": model(
                    url="https://github.com/alimbekovKZ/lungs_segmentation/releases/download/1.0.0/resnet34.pth",
                    model=Unet.Resnet(seg_classes=2, backbone_arch="resnet34"),
                ),
                "densenet121": model(
                    url="https://github.com/alimbekovKZ/lungs_segmentation/releases/download/1.0.0/densenet121.pth",
                    model=Unet.DensenetUnet(seg_classes=2, backbone_arch="densenet121"),
                ),
            }


            def create_model(model_name: str) -> nn.Module:
                model = models[model_name].model
                state_dict = model_zoo.load_url(
                    models[model_name].url, progress=True, map_location="cpu"
                )
                model.load_state_dict(state_dict)
                return model


            @st.cache(allow_output_mutation=True)
            def cached_model():
                model = create_model("resnet34")
                device = torch.device("cpu")
                model = model.to(device)
                return model


            model = cached_model()

            st.title("Segment lungs")
            def img_with_masks(img, masks, alpha, return_colors=False):
                """
                returns image with masks,
                img - numpy array of image
                masks - list of masks. Maximum 6 masks. only 0 and 1 allowed
                alpha - int transparency [0:1]
                return_colors returns list of names of colors of each mask
                """
                colors = [
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [255, 255, 0],
                    [0, 255, 255],
                    [102, 51, 0],
                ]
                color_names = ["Red", "greed", "BLue", "Yello", "Light", "Brown"]
                img = img - img.min()
                img = img / (img.max() - img.min())
                img *= 255
                img = img.astype(np.uint8)

                c = 0
                for mask in masks:
                    mask = np.dstack((mask, mask, mask)) * np.array(colors[c])
                    mask = mask.astype(np.uint8)
                    img = cv2.addWeighted(mask, alpha, img, 1, 0.0)
                    c = c + 1
                if return_colors is False:
                    return img
                else:
                    return img, color_names[0 : len(masks)]


            def inference(model, image, thresh=0.2):
                model.eval()
                image = (image - image.min()) / (image.max() - image.min())
                augs = aug(image=image)
                image = augs["image"].transpose((2, 0, 1))
                im = augs["image"]
                image = np.expand_dims(image, axis=0)
                image = torch.tensor(image)

                mask = torch.nn.Sigmoid()(model(image.float()))
                mask = mask[0, :, :, :].cpu().detach().numpy()
                mask = (mask > thresh).astype("uint8")
                return im, mask


            if image_file is not None:
                new_img = np.array(our_image.convert('RGB')) 
                image = cv2.cvtColor(new_img,1)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                st.image(gray, caption="Before", use_column_width=True)
                st.write("")
                st.write("Detecting lungs...")
                image, mask = inference(model, image, 0.2)
                st.image(
                    img_with_masks(image, [mask[0], mask[1]], alpha=0.1),
                    caption="Image + mask",
                    use_column_width=True,
                )
            


    if st.sidebar.button("About the Author"):
        st.sidebar.subheader("Chest X-Ray Tool")
        st.sidebar.markdown("by [GroupTuesday]")
        st.sidebar.markdown("[ngop2515@gmail.com]")
        st.sidebar.text("All Rights Reserved (2024)")
    
    st.sidebar.markdown("References")
    st.sidebar.markdown("https://github.com/Vinay10100/Chest-X-Ray-Classification/tree/main")
    st.sidebar.markdown("https://github.com/alimbekovKZ/lungs_segmentation")  


if __name__ == '__main__':
		main()	
			
			