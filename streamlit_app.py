import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.callbacks import get_openai_callback


# 텍스트 길이 측정을 위한 함수
def tiktoken_len(text):
    return len(text)


# 파일에서 텍스트를 추출하는 함수
def get_text(docs):
    """파일에서 텍스트를 읽어오는 함수"""
    all_texts = []

    if not docs:
        raise ValueError("파일이 업로드되지 않았습니다.")

    for file in docs:
        file_name = file.name
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file)
        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(file)
        elif file_name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(file)
        else:
            raise ValueError(f"지원되지 않는 파일 형식: {file_name}")

        documents = loader.load()
        if not documents:
            raise ValueError(f"{file_name}에서 텍스트를 추출할 수 없습니다.")

        for doc in documents:
            all_texts.append(doc.page_content)

    return all_texts


# 텍스트를 작은 조각으로 분할하는 함수
def get_text_chunks(texts):
    """텍스트를 작은 조각들로 분할"""
    if not texts or not isinstance(texts, list):
        raise ValueError("유효한 텍스트 데이터가 없습니다.")

    documents = [Document(page_content=text) for text in texts]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len,
    )

    return text_splitter.split_documents(documents)


# 벡터 저장소 생성
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store


# 대화 체인 구성
def get_conversation_chain(vector_store, api_key):
    chain = load_qa_chain(vector_store, api_key)
    return chain


# 메인 함수
def main():
    st.title("문서 처리 및 질문-응답 시스템")
    st.subheader("문서를 업로드하고 질문하세요.")

    # OpenAI API 키 입력
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    # 파일 업로드
    uploaded_files = st.file_uploader("파일 업로드", accept_multiple_files=True, type=["pdf", "docx", "pptx"])

    # 처리 버튼
    process = st.button("처리 시작")

    if process:
        try:
            # API 키 검증
            if not openai_api_key:
                st.info("OpenAI API 키를 입력해주세요.")
                return

            # 파일에서 텍스트 추출
            files_text = get_text(uploaded_files)
            if not files_text:
                st.warning("추출된 텍스트가 없습니다.")
                return

            # 텍스트 분할
            text_chunks = get_text_chunks(files_text)
            if not text_chunks:
                st.warning("텍스트 조각을 생성할 수 없습니다.")
                return

            # 벡터 저장소 생성
            vector_store = get_vectorstore(text_chunks)

            # 대화 체인 생성
            st.session_state.conversation = get_conversation_chain(vector_store, openai_api_key)
            st.session_state.processComplete = True

            st.success("처리가 완료되었습니다. 질문을 입력하세요.")

        except Exception as e:
            st.error(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
