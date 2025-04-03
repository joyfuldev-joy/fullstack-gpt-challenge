import streamlit as st
from langchain.prompts import PromptTemplate
from datetime import date, datetime

# streamlit 은 화면이 변경될때마다 python 코드를 재실행하면서 화면을 새로그림
thisTime = datetime.today().strftime("%H:%M:%S")
st.title(thisTime)

st.slider('Select a range of values', 0.0, 100.0, (25.0, 75.0))


