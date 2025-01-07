import tempfile
import streamlit as st

from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage

from doc_handler.llm.chat import LLMAgent


def init_agent():
    if 'agent' not in st.session_state:
        st.session_state.agent = LLMAgent()
        st.session_state.messages = []


def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def app():
    st.title("AI Assistant")
    init_agent()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="""Hello! How can I help?""")]

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload PDF (Optional)", type="pdf")
        if uploaded_file and 'current_file' not in st.session_state:
            with st.spinner("Processing PDF..."):
                pdf_path = save_uploaded_file(uploaded_file)
                st.session_state.agent.load_pdf(pdf_path)
                st.session_state.current_file = uploaded_file.name
                Path(pdf_path).unlink()
            st.success("PDF processed!")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    user_query = st.chat_input("Enter your query")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            with st.spinner("Thinking..."):
                # Get the response from the agent
                ai_response = st.session_state.agent.process_query(user_query)

                # Display the response in the chat
                st.write(ai_response)

                # Wrap the response in an AIMessage object and save it
                ai_message = AIMessage(content=ai_response)
                st.session_state.chat_history.append(ai_message)

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("""
        This AI assistant uses:
        - Open source LLMs for generating responses
        - RAG (Retrieval-Augmented Generation) for PDF knowledge
        - Tavily for web search when no PDF is provided
        """)

if __name__ == "__main__":
    app()