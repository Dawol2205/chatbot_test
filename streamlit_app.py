import streamlit as st
from openai import OpenAI
import numpy as np 

st.title("간단한 숫자 데이터 분석하기")

# 사용자로부터 숫자 입력받기
numbers = st.text_input("숫자 리스트를 입력하세요 (쉼표로 구분)", "1,2,3,4,5")  # 플레이스홀더, 기본값
number_list = [float(x) for x in numbers.split(",")]

# 통계 정보 계산
mean_value = np.mean(number_list)
median_value = np.median(number_list)
stdev_value = np.std(number_list)

# 결과 출력
st.write(f"평균값: {mean_value}")
st.write(f"중앙값: {median_value}")
st.write(f"표준편차: {stdev_value}")

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "응애가 만든 기본 템플릿입니다. gpt-3.5-turbo를 기준으로 생성하네요 "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
    "영어영어 쏼라쏼라"
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key 입력해주세요", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue. 빨리 입력하라구요.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
