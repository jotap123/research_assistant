import logging

from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

from doc_handler.config import chat
from doc_handler.utils import load_llm_chat
from doc_handler.llm.utils import ConversationState, ActionPlan, AgentConfig, RetrievalAction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.search = TavilySearchResults(max_results=self.config.max_search_results)
        self.llm = load_llm_chat(chat)
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        self.vectorstore = None
        self.graph = self.build_graph()


    def summarize_memory(self, state: ConversationState) -> ConversationState:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize the following conversation history:"),
                ("user", "{history}")
            ])
            chain = prompt | self.llm | StrOutputParser()
            history_text = "\n".join([msg["content"] for msg in state.memory[:-2]])
            summary = chain.invoke({"history": history_text})
            state.memory = state.memory[-2:]
            state.summary = summary
        except Exception as e:
            logging.error(f"Memory summarization failed: {e}")
            state.summary = "Summary generation failed"
        return state
    

    def should_continue(self, state: ConversationState) -> bool:
        """Determines whether memory summarization is needed."""
        return len(state.memory) > self.config.memory_summarizer_threshold


    def determine_action(self, state: ConversationState) -> ConversationState:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze the query and decide:
                - SEARCH: For general or current info
                - PDF: For document-specific queries
                - BOTH: To combine sources
                - NONE: For conversational queries."""),
                MessagesPlaceholder(variable_name="messages")
            ])
            parser = PydanticOutputParser(pydantic_object=ActionPlan)
            chain = prompt | self.llm | parser
            result = chain.invoke({"messages": state.memory})
            state.action_plan = result.dict()
        except Exception as e:
            logging.error(f"Action determination failed: {e}")
            state.action_plan = ActionPlan(
                action=RetrievalAction.NONE,
                reasoning="Error in action determination"
            ).model_dump()
        return state


    def load_pdf(self, pdf_file: str) -> None:
        try:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = splitter.split_documents(documents)
            self.vectorstore = Chroma.from_documents(chunks, self.embeddings)
            logging.info(f"Successfully loaded PDF: {pdf_file}")
        
        except Exception as e:
            logging.error(f"PDF loading failed: {e}")
            raise RuntimeError(f"Failed to load PDF: {e}")


    def retrieve_context(self, state: ConversationState) -> ConversationState:
        context_parts = []
        plan = ActionPlan.model_validate(state.action_plan)
        query = plan.search_query or state.memory[-1]["content"]

        try:
            if plan.action in [RetrievalAction.PDF, RetrievalAction.BOTH] and self.vectorstore:
                docs = self.vectorstore.similarity_search(query, k=3)
                context_parts.append("PDF Context:\n" + "\n".join([doc.page_content for doc in docs]))

            if plan.action in [RetrievalAction.SEARCH, RetrievalAction.BOTH]:
                web_results = self.search.invoke(query)
                context_parts.append("Web Search Results:\n" +
                                   "\n".join([f"{r['title']}: {r['content']}" for r in web_results]))
        except Exception as e:
            logging.error(f"Context retrieval failed: {e}")
            context_parts.append(f"Error retrieving context: {str(e)}")

        state.context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        return state


    def generate_response(self, state: ConversationState) -> ConversationState:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Use the provided context to respond.
                Context: {context}
                Previous Summary: {summary}"""),
                MessagesPlaceholder(variable_name="messages")
            ])
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "context": state.context,
                "summary": state.summary or "No previous summary",
                "messages": state.memory
            })
            state.memory.append(AIMessage(content=response))
        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            state.memory.append(AIMessage(content="I apologize, but I encountered an error generating a response."))
        return state


    def build_graph(self) -> StateGraph:
        """Build the enhanced LangGraph workflow."""
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("check_summary", self.should_continue)
        workflow.add_node("summarize", self.summarize_memory)
        workflow.add_node("plan", self.determine_action)
        workflow.add_node("retrieve", self.retrieve_context)
        workflow.add_node("generate", self.generate_response)

        # Add edges
        workflow.add_conditional_edges("check_summary", {True: "summarize", False: "plan"})
        workflow.add_edge("summarize", "plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Set entry point
        workflow.set_entry_point("check_summary")

        return workflow.compile()


    def process_query(self, query: str) -> str:
        try:
            self.state.memory.append(HumanMessage(content=query))
            result_state = self.graph.invoke(self.state)
            response = result_state.memory[-1].content
            self.state = result_state
            return response
        except Exception as e:
            logging.error(f"Query processing failed: {e}")
            return "I apologize, but I encountered an error processing your query."
