import os
import requests
import streamlit as st


st.set_page_config(page_title="Analog Matcher", layout="wide")
st.title("Поиск аналогов продукции")

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

with st.form("search"):
    name = st.text_input("Наименование")
    manufacturer = st.text_input("Производитель")
    article = st.text_input("Артикул")
    submitted = st.form_submit_button("Найти аналоги")

if submitted:
    try:
        resp = requests.post(
            f"{backend_url}/predict",
            json={"name": name, "manufacturer": manufacturer, "article": article},
            timeout=30,
        )
        data = resp.json()
        st.success("Найдено")
        st.write(data)
    except Exception as e:
        st.error(f"Ошибка: {e}")


