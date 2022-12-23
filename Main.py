import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
from streamlit_lottie import st_lottie

# 마스크 착용 감지 모델
# 기능 : 이용자로부터 얼굴 이미지를 입력받아, 모델을 통해 마스크를 썼는지 안 썼는지 감지 후 결과 출력

st.set_page_config(
    page_title="마스크 착용 감지 모델",
    page_icon="😷",
)

# @st.experimental_memo


@st.cache
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url = "https://assets6.lottiefiles.com/private_files/lf30_kbjokjdo.json"
# "https://assets8.lottiefiles.com/packages/lf20_uidhg9jw.json"
# "https://assets8.lottiefiles.com/packages/lf20_5q3dohib.json"
# "https://assets6.lottiefiles.com/private_files/lf30_kbjokjdo.json"

lottie_json = load_lottieurl(lottie_url)
st_lottie(lottie_json, speed=1, height=200, key="initial")

st.write('')
st.write('')

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.01, 1.7, 0.05, 1, 0.01)
)

with row0_1:
    st.title('😷 Mask or No Mask ?')


with row0_2:
    st.subheader('오9오9')
    st.write('''
    ***MINI project***  
    멋쟁이 사자처럼 AI SCHOOL 7기  
    권태윤, 김예지, 이정은, 임종우, 조예슬
    ''')

st.write('---')

# 제목
# st.header('😷 마스크 착용 감지 모델 🙂')

# 모델 임포트
# @st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('ResNet152V2_0.9659.h5')


model = load_model()


# 사진 입력받기
uploaded_file = st.file_uploader(
    "⬇️ 얼굴 사진을 올려주세요!", type=['png', 'jpg', 'jpeg'])

# 예측 및 결과 출력
height = 224
width = 224


@ st.cache
def title_predict(path):
    img = tf.keras.preprocessing.image.load_img(
        path, target_size=(height, width))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    pred = model.predict(np.array([img]))

    if pred[0][0] > 0.5:
        return f'Without Mask : {pred[0][0]*100 : 0.2f}%', 0
    else:
        return f'With Mask : {(1-pred[0][0])*100 : 0.2f}%', 1


if uploaded_file is not None:
    if title_predict(uploaded_file)[1] == 1:
        st.write('---')
        st.write('### 🙆‍♂️ 마스크를 착용하셨군요!')
    else:
        st.write('---')
        st.write('### ❌ 마스크를 착용하지 않으셨군요!')

    # img = cv2.cvtColor(cv2.imread(uploaded_file),cv2.COLOR_BGR2RGB)
    # img = cv2.imread(uploaded_file)
    # png error 발생 -> keras의 image 이용

    img = tf.keras.preprocessing.image.load_img(
        uploaded_file)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(img)

    ax.set_title(title_predict(uploaded_file)[0])
    plt.axis('off')
    st.pyplot(fig)
