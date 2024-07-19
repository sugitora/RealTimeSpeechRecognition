import streamlit as st

# デバッグ用のsecrets表示
st.write("Loaded secrets:", st.secrets)

# ネストされたキーへのアクセス
try:
    api_key = st.secrets["secrets"]["OPENAI_API_KEY"]
    st.write("API Key:", api_key)
except KeyError as e:
    st.write("KeyError:", e)
    st.write("Current secrets:", st.secrets)


# import streamlit as st

# try:
#     api_key = st.secrets["OPENAI_API_KEY"]
#     st.write("API Key:", api_key)
# except KeyError as e:
#     st.write("KeyError:", e)
#     st.write("Current secrets:", st.secrets)