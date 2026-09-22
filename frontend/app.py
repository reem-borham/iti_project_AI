"""
Textbook RAG Assistant — Streamlit Frontend
Phase 4 of the ITI AI Track Project.

Run with:
    streamlit run frontend/app.py
"""

import streamlit as st
import api_client

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Textbook RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sample benchmark queries (from Phase 2.6 evaluation)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What is a Python dictionary and how do you create one?",
    "Explain the difference between a list and a tuple in Python.",
    "What are Python decorators and when should you use them?",
    "How does exception handling work in Python? Give an example.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of overfitting and how to prevent it.",
    "What is a confusion matrix and what metrics can be derived from it?",
    "What is exploratory data analysis (EDA) and why is it important?",
    "What is cross-validation and why is it preferred over a simple train/test split?",
]

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Assistant")
    st.caption("ITI AI Track — Textbook Q&A")
    st.divider()

    # ── Backend Health ──────────────────────────────────────────────────────
    st.subheader("🔌 Backend Status")
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.cache_data.clear()

    health = api_client.check_health()

    if health.get("ok"):
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Offline")
        if health.get("error"):
            st.caption(f"Error: {health['error']}")

    col1, col2 = st.columns(2)
    with col1:
        if health.get("vector_store_loaded"):
            st.metric("Vector Store", "✅ Ready")
        else:
            st.metric("Vector Store", "⚠️ Empty")
    with col2:
        if health.get("ollama_connected"):
            st.metric("Ollama LLM", "✅ Ready")
        else:
            st.metric("Ollama LLM", "⚠️ Offline")

    st.caption(f"Model: `{health.get('model', '—')}`")
    st.caption(f"Collection: `{health.get('collection', '—')}`")

    st.divider()

    # ── Sample Queries ──────────────────────────────────────────────────────
    st.subheader("💡 Quick Questions")
    st.caption("Click any question to ask it instantly:")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"sample_{hash(q)}"):
            st.session_state.pending_question = q

    st.divider()

    # ── Clear Chat ──────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()
    st.caption(
        "**Textbooks indexed:**\n"
        "- Introduction to Python Programming\n"
        "- Principles of Data Science"
    )
    st.caption("Answers cite source pages from these textbooks.")

# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────
st.title("📚 Textbook RAG Assistant")
st.caption(
    "Ask any question about **Python Programming** or **Data Science** — "
    "answers are grounded in your textbooks with page-level citations."
)

# Warn if backend is offline before user tries
if not health.get("ok"):
    st.warning(
        "⚠️ The backend is not reachable. Make sure the FastAPI server is running:\n\n"
        "```bash\ncd backend\nuvicorn app.main:app --reload\n```",
        icon="⚠️",
    )

if not health.get("vector_store_loaded"):
    st.info(
        "ℹ️ The vector store appears empty. Run the RAG pipeline notebook first:\n\n"
        "`notebooks/rag_pipeline.ipynb` → re-run all cells to index the textbooks.",
        icon="ℹ️",
    )

st.divider()

# ── Display existing conversation ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources cited", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# ─────────────────────────────────────────────────────────────────────────────
# Handle input — either from chat box or a sidebar quick-question button
# ─────────────────────────────────────────────────────────────────────────────
def handle_question(question: str) -> None:
    """Run the RAG query and append messages to session state."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    # Query backend
    with st.chat_message("assistant"):
        with st.spinner("Searching textbooks and generating answer…"):
            try:
                result = api_client.query_assistant(question)
                answer = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])

                st.markdown(answer)
                if sources:
                    with st.expander("📄 Sources cited", expanded=True):
                        for src in sources:
                            st.markdown(f"- `{src}`")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except RuntimeError as exc:
                err_msg = str(exc)
                st.error(f"❌ {err_msg}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ Error: {err_msg}", "sources": []}
                )


# Process sidebar quick-question button
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    handle_question(question)

# Process chat input
user_input = st.chat_input("Ask a question about Python or Data Science…")
if user_input:
    handle_question(user_input)
