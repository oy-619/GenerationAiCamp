import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv()

st.title("LLM機能を搭載したWebアプリ")

st.write("##### 動作モード: 料理の専門家")
st.write("入力フォームにテキストを入力し、「実行」ボタンを押すことで選択した料理の専門家の回答が得られます。")
st.write("##### 動作モード: 野球の専門家")
st.write("入力フォームにテキストを入力し、「実行」ボタンを押すことで選択した野球の専門家の回答が得られます。")

selected_item = st.radio(
    "質問する専門家を選択してください。",
    ["料理の専門家", "野球の専門家"]
)

st.divider()

input_message = st.text_input(label="質問を入力してください。")

if st.button("実行"):
    st.divider()

    if not input_message:
        st.error("テキストを入力してから「実行」ボタンを押してください。")
        exit
    
    if selected_item == "料理の専門家":
        content="あなはたは料理の専門家です。"  
    else:
        content="あなはたは野球の専門家です。"

    messages = [
        SystemMessage(content=content),
        HumanMessage(content=input_message),
    ]

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, verbose=True)
    print(llm)
    result = llm(messages)
    st.write(f"回答: **{result.content}**")