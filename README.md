# AutoStream Social-to-Lead Agent

AutoStream's conversational AI agent handles user engagements by parsing intent, answering product/pricing questions using local Retrieval-Augmented Generation (RAG), and capturing high-intent leads gracefully when users express willingness to try or buy.

## 1. How to run the project locally

**Prerequisites:**
- Python 3.9+
- A Google API key (for Gemini 1.5 Flash LLM)

**Steps:**
1. Clone the repository and navigate into the project directory.
2. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Google API key:
   * **Windows:** `set GOOGLE_API_KEY=your_key_here`
   * **Mac/Linux:** `export GOOGLE_API_KEY=your_key_here`
4. Run the conversational AI:
   ```bash
   python agent.py
   ```
5. You can test the following conversational flow:
   - *User:* "Hi there!"
   - *User:* "What's the pricing like for Pro?"
   - *User:* "Hmm that sounds good, I'd like to try it for my channel."
   - *(Agent asks for details)*
   - *User:* "My name is Alice."
   - *User:* "My email is alice@example.com."
   - *User:* "I make YouTube videos."
   - *(Agent triggers tool successfully)*

---

## 2. Architecture Explanation

**LangGraph Choice**: I chose LangGraph because building complex, stateful agents requires granular control over execution flow rather than relying on abstract agent loops. LangGraph allows explicitly defining nodes (like intent classification, RAG lookup, and lead capture) and deterministic conditional edges to route the user precisely based on context. This graph-based state machine reduces hallucinations, enforces strict guardrails around tool execution, and natively supports dynamic cyclic interactions required when probing users step-by-step for missing lead info.

**State Management**: The application state (`AgentState`) is structured around a central `TypedDict` retaining a chronological message history (`messages` with the `add_messages` reducer). Additionally, it stores domain-specific keys like `intent`, `lead_name`, `lead_email`, `lead_platform`, and `tool_executed`. Crucially, state persistance across the 5-6 multi-turn conversation is managed by injecting LangGraph's `MemorySaver` checkpointer into the compiled graph configuration (`thread_id="1"`). With every user input, the state machine restores context from memory, allowing the intent classifier to deeply understand progressive intent shifts (e.g., transitioning from an inquiry to a high-intent signup).

---

## 3. WhatsApp Deployment Webhook Strategy

To deploy this agent logic on WhatsApp using the official API/Webhooks:
1. **Webhook Endpoint Handler**: We would wrap our LangGraph agent in a lightweight ASGI framework like FastAPI. We create a `POST /webhook` endpoint that Facebook/Meta forwards WhatsApp messages to.
2. **Session / Thread Management**: Currently, the CLI uses a hardcoded `thread_id`. For WhatsApp, we'd use the sender's mobile number (`From` field in the webhook payload) as the `thread_id`. The LangGraph Checkpointer (e.g., backed by Postgres or Redis in production, rather than in-memory) will associate that number with their session state.
3. **Async Job Processing**: The FastAPI endpoint should acknowledge the webhook immediately (`HTTP 200 OK`) and pass the incoming message to a background worker queue (like Celery).
4. **Execution & Reply**: The worker invoked the LangGraph with the user's message. Upon the agent yielding a response, the backend triggers an outgoing HTTP `POST` request to the WhatsApp Cloud API (`messages` endpoint) to deliver the generated reply to the user's phone.
