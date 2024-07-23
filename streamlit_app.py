import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages.base import BaseMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


st.set_page_config(page_title="Personal Digital Library Assistant")
st.title("Personal Digital Library Assistant")


@st.cache_resource
def get_retriever():
    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=os.getenv("MONGODB_URI"),
        namespace="books.unstructured-demo",
        embedding=HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL")),
        text_key="text",
        embedding_key="embeddings",
    )

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def get_question(input):
    if not input:
        return None
    elif isinstance(input, str):
        return input
    elif isinstance(input, dict) and 'question' in input:
        return input['question']
    elif isinstance(input, BaseMessage):
        return input.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain():
    retriever = get_retriever()

    local_model = "llama3" # switch to llama3b
    model = ChatOllama(model=local_model,
                       num_predict=400,
                       stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"])

    system_prompt = """
    <|start_header_id|>user<|end_header_id|>
    You are a helpful and knowledgeably AI Librarian. Use the following context and the users' chat history to 
    help the user. If you don't know the answer, just say that you don't know.  
    Context: {context}
    Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
            {
                "context": RunnableLambda(get_question) | retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | model
    )

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    runnable = contextualize_q_prompt | model | rag_chain
    chat_memory = ChatMessageHistory()

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return with_message_history


def ask_question(chain, query):
    response = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )
    return response


def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Flipping pages..."):
                response = ask_question(qa, prompt)
                st.markdown(response.content)
        message = {"role": "assistant", "content": response.content}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    load_dotenv()

    chain = get_chain()
    st.subheader("Ask me questions about your digital library")
    show_ui(chain, "What would you like to know?")
