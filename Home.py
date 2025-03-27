import streamlit as st
from langchain.prompts import PromptTemplate

st.title("Home")
st.header("Welcome to the Home Page")
st.markdown("""
This is a simple web application that demonstrates the use of Streamlit.
- Use the sidebar to navigate to the different pages.
- The pages include:
    - Home
    - Data Analysis
    - Data Visualization
    - Machine Learning
    - About Us      
            """)


# streamlit 은 내가 원하는 것을 최대한 출력해주려한다. 심지어 class 를 출력하면 method 및 주석까지도 보여줌. 
# magic : write method 를 사용하지 않고도 화면에 출력이 가능하다.
st.write("hello")
st.write([1,2,3,4])
st.write({"name":"John", "age":30})
st.write(PromptTemplate)

p = PromptTemplate(input_variables=["foo"], template="Say {foo}")
st.write(p)
