import streamlit as st
import os
from tempfile import NamedTemporaryFile
from doc_handler.llm.chat import LLMAgent  # Import the LLMAgent class from previous code


def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    if uploaded_file is not None:
        # Create a temporary file
        with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    return None


def app():
    st.title("🤖 AI Assistant with PDF Knowledge")
    st.write("""
    This AI assistant can answer questions based on uploaded PDF documents or search the internet if no PDF is provided.
    """)

    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Initialize agent
    agent = LLMAgent()

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document (optional)", type="pdf")
    
    if uploaded_file:
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                # Save uploaded file and load it into the agent
                pdf_path = save_uploaded_file(uploaded_file)
                agent.load_pdf(pdf_path)
                st.session_state.current_file = uploaded_file.name
                os.unlink(pdf_path)  # Clean up temporary file
            st.success("PDF processed successfully!")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question"):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.invoke(prompt)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("""
        This AI assistant uses:
        - GPT-4 for generating responses
        - RAG (Retrieval-Augmented Generation) for PDF knowledge
        - Tavily for web search when no PDF is provided
        """)
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    app()