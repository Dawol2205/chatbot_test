import streamlit as st
from openai import OpenAI
import re

# ---------------------------------------------------------------------------------------
# Set page config
st.set_page_config(page_title="🥘 한식 레시피 도우미", page_icon="🥘")

# Show title and description
st.title("🥘 한식 레시피 도우미")
st.write(
    "한식 레시피를 AI와 함께 알아보세요! 요리 방법, 재료, 팁 등을 물어보실 수 있습니다. "
    "음식 이름을 말씀해 주시면 상세한 레시피를 알려드립니다."
)

# System message to guide the AI's behavior
SYSTEM_MESSAGE = """당신은 한식 전문 요리 선생님입니다. 네이버 백과사전의 한식 데이터를 기반으로 요리 방법을 알려주세요.
답변할 때는 다음 형식을 따라주세요:

1. 요리 설명: 간단한 소개와 특징
2. 필수 재료: bullet point로 재료 나열
3. 선택 재료: bullet point로 선택적인 재료 나열
4. 조리 순서: 번호를 매겨 순서대로 설명
5. 조리 팁: 맛있게 만들기 위한 중요 포인트

답변은 친근하고 이해하기 쉬운 말투로 해주세요."""

# Initialize OpenAI client
openai_api_key = st.text_input("OpenAI API Key를 입력해주세요", type="password")
if not openai_api_key:
    st.info("OpenAI API 키를 입력해주세요! API 키는 안전하게 보관됩니다.", icon="🔑")
else:
    client = OpenAI(api_key=openai_api_key)

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_MESSAGE}
        ]

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] != "system":  # Don't show system messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("어떤 요리를 배워볼까요? (예: 김치찌개 레시피 알려주세요)"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Add sidebar with helpful tips
    with st.sidebar:
        st.header("💡 사용 팁")
        st.markdown("""
        - 특정 요리의 레시피를 물어보세요
        - 재료 대체 방법을 문의하세요
        - 조리 팁을 요청해보세요
        - 칼로리나 영양 정보도 물어볼 수 있어요
        """)
        
        st.header("🎯 예시 질문")
        st.markdown("""
        - "김치찌개 레시피 알려주세요"
        - "불고기 양념 비율이 궁금해요"
        - "된장찌개에 들어가는 재료가 뭔가요?"
        - "비빔밥 예쁘게 담는 방법 알려주세요"
        """)
