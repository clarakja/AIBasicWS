#이전 수업 시간에 만들었던 이미지 분류 pkl 파일을 바탕으로 한 이미지 분류 모델을 Streamlit에 올리는 예제 코드
#파일 이름 streamlit_app.py

import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1NKIhMhUeRC0vPptHwT4it-LMYhamVDyi'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

# 모델의 분류 라벨 출력
labels = learner.dls.vocab
#st.write(labels)
st.title(f"이미지 분류기 (Fastai) - 분류 라벨: {', '.join(labels)}")

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    # 업로드된 이미지 보여주기
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # Fastai에서 예측을 위해 이미지를 처리
    img = PILImage.create(uploaded_file)

    # 예측 수행
    prediction, _, probs = learner.predict(img)

    # 결과 출력
    st.write(f"예측된 클래스: {prediction}")


    # 클래스별 확률을 HTML과 CSS로 시각화
    st.markdown("<h3>클래스별 확률:</h3>", unsafe_allow_html=True)

    # if prediction == labels[0]:
    #     st.write("중냉 꿋굿")
    # elif prediction == labels[1]:
    #     st.write("짜장면은 굿")
    # elif prediction == labels[2]:
    #     st.write("짬뽕은 맵지만 맛있어!!")

    for label, prob in zip(labels, probs):
        # HTML 및 CSS로 확률을 시각화
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


