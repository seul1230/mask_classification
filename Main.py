import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# 마스크 착용 감지 모델
# 기능 : 이용자로부터 얼굴 이미지를 입력받아, 모델을 통해 마스크를 썼는지 안 썼는지 감지 후 결과 출력

st.set_page_config(
    page_title="마스크 착용 감지 모델",
    page_icon="😷",
)

st.header("""오9오9  
***MINI project***  
멋쟁이 사자처럼 AI SCHOOL 7기  
권태윤, 김예지, 이정은, 임종우, 조예슬

---
""")

# 제목
st.title('😷 마스크 착용 감지 모델 🙂')

# 모델 임포트
model = tf.keras.models.load_model('ResNet152V2_0.9659.h5')
# model.summary()

# 사진 입력받기
uploaded_file = st.file_uploader("얼굴 사진을 올려주세요!", type=['png', 'jpg', 'jpeg'])

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


# def predict(path):
#     img = tf.keras.preprocessing.image.load_img(
#         path, target_size=(height, width), interpolation='lanczos')
#     img = tf.keras.preprocessing.image.img_to_array(img)
#     img = img / 255.0
#     pred = model.predict(np.array([img]))
#     plt.imshow(img)
#     if pred[0][0] > 0.5:
#         st.write('## 🙆‍♂️ 마스크를 착용하셨군요!')
#         plt.title(f'Without Mask : {pred[0][0]*100 : 0.2f}%')
#     else:
#         st.write('## ❌ 마스크를 착용하지 않으셨군요!')
#         plt.title(f'With Mask : {(1-pred[0][0])*100 : 0.2f}%')


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
    plt.rc('font', size=15)
    plt.axis('off')
    st.pyplot(fig)
