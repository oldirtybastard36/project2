import streamlit as st

from src.model import model
from src.utils import clean_text

if "text" not in st.session_state:
    st.session_state["text"] = ""

st.write("""
# Определяет токчисность текста
Введите определенное сообщение в поле ниже и нажмите Enter. После нейросеть определит, токсично ли оно
""")


def execute_model(input):
    result = model.predict([clean_text(input)])
    return result


def function():
    text: str = st.session_state.data
    text = text.strip()
    toxicity = execute_model(text)

    text = "Токсично" if toxicity else "Безопасно"
    st.session_state.text = text


st.text_area("Введите текст: ", key="data", on_change=function)
st.text_input("Результат:", key="text", disabled=True)
