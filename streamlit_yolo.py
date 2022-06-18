import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import streamlit as st
from imageai.Detection import ObjectDetection
from PIL import Image

execution_path = os.getcwd()

@st.cache
def load_image(img):
    im = Image.open(img)
    return im

@st.cache(allow_output_mutation=True)
def load_models(model_file):
    prediction = ObjectDetection()
    prediction.setModelTypeAsYOLOv3()
    prediction.setModelPath(model_file)
    prediction.loadModel()
    return prediction

prediction = load_models("./models/yolo.h5")

def main():
    st.title("Object Detection App")
    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file is not None:
        input_image = Image.open(image_file)
        st.text("Original Image")
        st.image(input_image)
        returned_image, detections = prediction.detectObjectsFromImage(input_image=input_image,
                                                                       input_type='array',
                                                                       output_type='array')
        for eachObject in detections:
            st.write(eachObject["name"], " : ", eachObject["percentage_probability"])
        st.image(returned_image)

if __name__ == '__main__':
    main()
