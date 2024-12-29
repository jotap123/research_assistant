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
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': 'cuda'})
        self.vectorstore = None
        self.state = ConversationState(memory=[])
        self.graph = self.build_graph()

        self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


    def determine_action(self, state: ConversationState) -> ConversationState:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Based on the conversation history and context, determine the best action:
                - SEARCH: For current or general information needs
                - NONE: For conversational responses
                
                Provide the action with a very brief explanation.
                Previous Context: {summary}"""),
                MessagesPlaceholder(variable_name="messages")
            ])
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "messages": [msg.dict() for msg in state.memory],
                "summary": state.summary or "No previous context available."
            })

            # Validate and store the generated action plan in the state.
            if "SEARCH" in result.upper():
                state['action_plan'].action = RetrievalAction.SEARCH
            elif "NONE" in result.upper():
                state['action_plan'].action = RetrievalAction.NONE

            logging.info(f"Determined action: {state['action_plan'].action}")
        except Exception as e:
            logging.error(f"Action determination failed: {e}")
            state.action_plan = ActionPlan(
                action=RetrievalAction.ERROR,
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
        query = state.memory[-1]["content"]
        
        # Enhance query with conversation context
        if state.summary:
            query_terms = set(query.lower().split())
            summary_terms = set(state.summary.lower().split())
            relevant_context = " ".join(term for term in summary_terms if term in query_terms)
            enhanced_query = f"{query} {relevant_context}".strip()
        else:
            enhanced_query = query

        try:
            if state['action_plan'].action == RetrievalAction.SEARCH and self.vectorstore:
                docs = self.vectorstore.similarity_search(enhanced_query, k=3)
                # Filter for relevance
                relevant_docs = [
                    doc for doc in docs 
                    if any(term in doc.page_content.lower() for term in query.lower().split())
                ]
                if relevant_docs:
                    context_parts.append("PDF Context:\n" + "\n".join(
                        [doc.page_content for doc in relevant_docs[:2]]
                    ))

            else:
                web_results = self.search.invoke(enhanced_query)
                filtered_results = [
                    r for r in web_results 
                    if any(term in r['content'].lower() for term in query.lower().split())
                ]
                if filtered_results:
                    context_parts.append("Web Search Results:\n" + "\n".join(
                        [f"{r['title']}: {r['content']}" for r in filtered_results[:2]]
                    ))

        except Exception as e:
            logging.error(f"Context retrieval failed: {e}")
            context_parts.append(f"Error retrieving context: {str(e)}")

        state.context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        return state


    def generate_response(self, state: ConversationState) -> ConversationState:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Generate a helpful response using the available information.
                If the context is relevant, incorporate it naturally.
                If information is missing, acknowledge limitations.
                
                Available Context: {context}
                Previous Summary: {summary}"""),
                MessagesPlaceholder(variable_name="messages"),
                ("system", "Provide a clear, natural response that addresses the query directly.")
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
            state.memory.append(
                AIMessage(content="I apologize, but I encountered an error generating a response.")
            )
        return state


    def should_continue(self, state: ConversationState) -> bool:
        """Determines whether memory summarization is needed."""
        if len(state.memory) > self.config.memory_summarizer_threshold:
            return "summarize"
        else:
            return END 


    def summarize_memory(self, state: ConversationState) -> ConversationState:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Create a concise yet informative summary of the conversation history.
                Include:
                - Main topics and questions discussed
                - Key information or preferences shared
                - Important context for future responses"""),
                ("user", "{history}")
            ])
            chain = prompt | self.llm | StrOutputParser()
            history_text = "\n".join([
                f"{'User' if i%2==0 else 'Assistant'}: {msg['content']}" 
                for i, msg in enumerate(state.memory[:-2])
            ])
            summary = chain.invoke({"history": history_text})
            state.memory = state.memory[-2:]
            state.summary = summary
        except Exception as e:
            logging.error(f"Memory summarization failed: {e}")
            state.summary = "Summary generation failed"
        return state


    def build_graph(self) -> StateGraph:
        """Build the enhanced LangGraph workflow."""
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("plan", self.determine_action)
        workflow.add_node("retrieve", self.retrieve_context)
        workflow.add_node("generate", self.generate_response)
        workflow.add_node("summarize", self.summarize_memory)

        # Add edges
        workflow.add_conditional_edges(
            "plan", lambda x: "retrieve" if x['action_plan'].action == RetrievalAction.SEARCH else "END",
            {"retrieve": "retrieve", "END": END}
        )
        workflow.add_edge("retrieve", "generate")
        workflow.add_conditional_edges("generate", self.should_continue)
        workflow.add_edge("summarize", END)

        workflow.set_entry_point("plan")

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
