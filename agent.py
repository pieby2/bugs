import os
import sys
from typing import TypedDict, Annotated, Literal
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from rich.console import Console
from langgraph.checkpoint.memory import MemorySaver

console = Console()

# --- Tool Execution ---
def mock_lead_capture(name: str, email: str, platform: str):
    """Mocks backend logic to capture a lead."""
    print(f"\n[LEAD CAPTURED SUCCESSFULLY]: Name: {name}, Email: {email}, Platform: {platform}")
    return True

# --- State Definition ---
class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None
    tool_executed: bool

class IntentClassification(BaseModel):
    intent: Literal["casual_greeting", "product_inquiry", "high_intent_lead"] = Field(
        description="Classify the user intent into one of the three categories based on the conversation history."
    )

class LeadExtraction(BaseModel):
    name: str = Field(description="The user's name if provided, else empty string.", default="")
    email: str = Field(description="The user's email if provided, else empty string.", default="")
    platform: str = Field(description="The user's creator platform (YouTube, Instagram, etc) if provided, else empty string.", default="")

def build_agent(api_key: str, model_name: str = "llama3-8b-8192"):
    """Builds and compiles the LangGraph app with Groq LLM initialized with the given key."""
    # Initialize LLM
    llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name=model_name)

    # Lightweight local embeddings to avoid external embedding limits
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        loader = TextLoader("knowledge.md")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        console.print(f"[bold red]Error initializing RAG: {e}[/bold red]")
        retriever = None

    def classify_intent(state: AgentState):
        system_prompt = (
            "You are an intent classifier for AutoStream, a video editing SaaS.\n"
            "Analyze the user's latest message and the conversation context to determine their intent:\n"
            "1. casual_greeting: User is merely saying hi or starting a casual chat without questions.\n"
            "2. product_inquiry: User is asking about pricing, features, or policies of the SaaS.\n"
            "3. high_intent_lead: User expresses a desire to buy, try, or sign up for a plan.\n"
            "Only output the intent label. Do not explain."
        )
        # Some LLMs via Groq might require formatting workarounds, but with_structured_output works usually.
        structured_llm = llm.with_structured_output(IntentClassification)
        messages_to_pass = [SystemMessage(content=system_prompt)] + state["messages"]
        result = structured_llm.invoke(messages_to_pass)
        return {"intent": result.intent}

    def handle_greeting(state: AgentState):
        system_prompt = (
            "You are an AI assistant for AutoStream. Respond casually and politely to the user's greeting.\n"
            "Ask how you can help them with their video editing needs today."
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def handle_inquiry(state: AgentState):
        latest_message = state["messages"][-1].content
        if retriever:
            docs = retriever.invoke(latest_message)
            context = "\n".join([doc.page_content for doc in docs])
        else:
            context = "Knowledge base unavailable."

        system_prompt = (
            f"You are a helpful assistant for AutoStream. Answer the user's question using ONLY the context provided below.\n"
            f"Context:\n{context}\n\n"
            f"If the answer is not in the context, politely say you don't have that information. Do not invent answers."
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def handle_lead(state: AgentState):
        if state.get("tool_executed"):
            response = AIMessage(content="You are all set! We will be in touch shortly.")
            return {"messages": [response]}

        system_prompt = (
            "Extract the user's name, email, and creator platform (like YouTube, Instagram, Twitch, etc.) "
            "from the conversation. If a piece of information is missing, leave it as an empty string."
        )
        
        structured_llm = llm.with_structured_output(LeadExtraction)
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        extracted = structured_llm.invoke(messages)

        name = extracted.name if extracted.name else state.get("lead_name", "")
        email = extracted.email if extracted.email else state.get("lead_email", "")
        platform = extracted.platform if extracted.platform else state.get("lead_platform", "")
        
        missing = []
        if not name: missing.append("name")
        if not email: missing.append("email")
        if not platform: missing.append("creator platform (e.g., YouTube, Instagram)")

        if missing:
            missing_str = ", ".join(missing)
            msg_content = f"I'd love to help you get started with AutoStream! To proceed, I just need a bit more information: {missing_str}."
            response = AIMessage(content=msg_content)
            return {
                "messages": [response],
                "lead_name": name,
                "lead_email": email,
                "lead_platform": platform
            }
        else:
            mock_lead_capture(name, email, platform)
            response = AIMessage(content="Perfect! Your details have been captured successfully. Our team will contact you shortly!")
            return {
                "messages": [response],
                "lead_name": name,
                "lead_email": email,
                "lead_platform": platform,
                "tool_executed": True
            }

    def route_intent(state: AgentState):
        intent = state.get("intent")
        if intent == "casual_greeting":
            return "greeting"
        elif intent == "product_inquiry":
            return "inquiry"
        elif intent == "high_intent_lead":
            return "lead"
        return "greeting"

    workflow = StateGraph(AgentState)
    workflow.add_node("classify", classify_intent)
    workflow.add_node("greeting", handle_greeting)
    workflow.add_node("inquiry", handle_inquiry)
    workflow.add_node("lead", handle_lead)

    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges("classify", route_intent)

    workflow.add_edge("greeting", END)
    workflow.add_edge("inquiry", END)
    workflow.add_edge("lead", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# Disable CLI execution at bottom for now, as we mainly run via Streamlit.
if __name__ == "__main__":
    if "GROQ_API_KEY" not in os.environ:
        console.print("[bold red]Please set GROQ_API_KEY environment variable.[/bold red]")
        sys.exit(1)
        
    app = build_agent(os.environ["GROQ_API_KEY"])
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]: break
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for state_update in app.stream(inputs, config):
            for node_name, values in state_update.items():
                if node_name in ["greeting", "inquiry", "lead"]:
                    console.print(f"[bold blue]Agent:[/bold blue] {values['messages'][-1].content}")
