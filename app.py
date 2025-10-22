# app.py
import os
import io
import pandas as pd
import streamlit as st

# LangChain + Groq imports
try:
    from langchain_groq import ChatGroq
except Exception:
    # fallback import name if package differs in env
    from langchain_groq import ChatGroq  # will raise if not installed

# CSV agent helpers (create_csv_agent lives in langchain_experimental or langchain_experimental.agents)
try:
    from langchain_experimental.agents import create_csv_agent
except Exception:
    try:
        from langchain.agents import create_csv_agent
    except Exception:
        create_csv_agent = None

# Optional RAG / Embeddings / Vectorstore
try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
except Exception:
    OpenAIEmbeddings = None
    FAISS = None
    RetrievalQA = None

# Memory
try:
    from langchain.memory import ConversationBufferMemory
except Exception:
    # Newer LangChain might have different path; handle gracefully
    try:
        from langchain.chains.conversation.memory import ConversationBufferMemory
    except Exception:
        ConversationBufferMemory = None

# Utilities
from typing import Optional

st.set_page_config(page_title="Ask your CSV (Groq version)", layout="wide")

APP_TITLE = "Ask your CSV (Groq version)"
st.title(APP_TITLE)
st.markdown("Upload a CSV, ask questions, and get answers from Groq models via LangChain. ✅")

# Sidebar - settings
st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox(
    "Select Groq model (suggested)",
    options=[
        "openai/gpt-oss-120b",
        "mixtral-8x7b-32768",
        "llama3-8b",
        "compound-beta",
        "deepseek-r1-distill-llama-70b",
    ],
    index=0,
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
max_tokens = st.sidebar.number_input("Max tokens (None = model default)", min_value=64, max_value=65536, value=2048, step=64)

enable_memory = st.sidebar.checkbox("Enable conversation memory (dev-mode)", value=True)
enable_rag = st.sidebar.checkbox("Enable RAG (use embeddings + retriever)", value=False)

st.sidebar.markdown("---")
st.sidebar.write("Environment variables required:")
st.sidebar.write("- `GROQ_API_KEY` (required)")
st.sidebar.write("- `OPENAI_API_KEY` (optional; required only if you enable RAG embeddings using OpenAI embeddings)")
st.sidebar.markdown("See README notes at bottom of this page for install & run commands.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="CSV containing OHLC or other trading data works fine.")

# Small helper to create LLM
def create_llm(model_name: str, temperature: float = 0.0, max_tokens: Optional[int] = None):
    if "GROQ_API_KEY" not in os.environ:
        st.error("GROQ_API_KEY not found in environment. Set it before running.")
        st.stop()
    # ChatGroq instantiation (from langchain_groq)
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens else None,
        # You can pass more kwargs here (timeout, reasoning_format, etc.)
    )
    return llm

# Display CSV summary
def summarize_csv(df: pd.DataFrame):
    st.subheader("CSV preview & summary")
    st.dataframe(df.head(20))
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.markdown("### Quick auto-summary (first rows / numeric stats)")
    st.write(df.describe(include='all').transpose())

# Build retriever if RAG
def build_retriever_from_df(df: pd.DataFrame):
    if OpenAIEmbeddings is None or FAISS is None:
        st.error("RAG dependencies not installed or not available (OpenAIEmbeddings / FAISS). See requirements.")
        return None
    # Convert dataframe rows into a list of docs (simple)
    texts = []
    METADATA_COLS = list(df.columns[:3]) if df.shape[1] >= 3 else list(df.columns)
    for i, row in df.iterrows():
        # We create a small textual representation for each row
        rdict = row.to_dict()
        text = " | ".join([f"{k}: {v}" for k, v in rdict.items()])
        texts.append({"text": text, "meta": {"row_index": i}})
    # Build documents list for embeddings
    docs = [t["text"] for t in texts]
    # Build embeddings (OpenAI Embeddings used here as option)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(docs, embeddings, metadatas=[t["meta"] for t in texts])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever

# Main interaction block
if uploaded_file is not None:
    # Read CSV to DataFrame (handle encodings)
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        # try alternative encodings
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        try:
            df = pd.read_csv(io.StringIO(raw.decode('latin-1')))
        except Exception as e2:
            st.error(f"Failed to read CSV: {e} / {e2}")
            st.stop()

    summarize_csv(df)

    st.markdown("---")
    st.subheader("Agent settings & run")
    col1, col2 = st.columns([2, 1])

    with col1:
        user_question = st.text_input("Ask a question about your CSV (e.g., 'What is the highest close?' or 'Summarize last 30 days'):", value="")
    with col2:
        run_button = st.button("Ask Groq")

    # Optionally: quick summarizer button
    if st.button("Auto summarise CSV (quick)"):
        # Create a short prompt to summarize first/last rows + numeric columns
        sample_text = df.head(10).to_csv(index=False)
        prompt = f"Summarize the CSV sample below (what columns, any obvious anomalies or trends). CSV:\n\n{sample_text}"
        llm = create_llm(model_choice, temperature=temperature, max_tokens=max_tokens)
        ai = llm.invoke([("system", "You are a helpful assistant that summarizes CSV data."),
                         ("human", prompt)])
        st.write(ai.content if hasattr(ai, "content") else ai)

    # Prepare LLM & memory
    llm = create_llm(model_choice, temperature=temperature, max_tokens=max_tokens)
    memory = None
    if enable_memory:
        if ConversationBufferMemory is None:
            st.warning("ConversationBufferMemory not available in this LangChain installation.")
        else:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Build retriever if user asked for RAG
    retriever = None
    if enable_rag:
        with st.spinner("Building retriever (embeddings + index). This may take a moment..."):
            retriever = build_retriever_from_df(df)
            if retriever:
                st.success("Retriever ready. Using RAG for question answering.")

    # Core: handle user ask
    if run_button and user_question.strip() != "":
        with st.spinner("Thinking with Groq..."):
            try:
                # If RAG retriever present -> use RetrievalQA
                if retriever is not None and RetrievalQA is not None:
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
                    out = qa_chain.run(user_question)
                    st.markdown("### Answer (RAG)")
                    st.write(out)
                else:
                    # Try using create_csv_agent if available for direct dataframe querying
                    if create_csv_agent is not None:
                        # create_csv_agent expects path or file-like. Our uploaded_file is file-like; ensure pointer at start
                        uploaded_file.seek(0)
                        # Save a temp local file to pass path (safer)
                        tmp_path = "/tmp/streamlit_uploaded.csv"
                        df.to_csv(tmp_path, index=False)
                        agent = create_csv_agent(llm, tmp_path, verbose=False)
                        # run agent
                        # Some agent executors expect run() or invoke(); handle both
                        try:
                            result = agent.run(user_question)
                        except Exception:
                            try:
                                response = agent.invoke({"input": user_question})
                                result = response.get("output") if isinstance(response, dict) else response
                            except Exception as e:
                                result = f"Agent error: {e}"
                        st.markdown("### Answer (CSV Agent)")
                        st.write(result)
                    else:
                        # Fallback: simple prompt with sample rows + question
                        sample = df.head(20).to_csv(index=False)
                        prompt = [
                            ("system", "You are a helpful data analyst."),
                            ("human", f"I have this CSV (sample):\n\n{sample}\n\nAnswer this question based on the CSV: {user_question}")
                        ]
                        ai_msg = llm.invoke(prompt)
                        answer = ai_msg.content if hasattr(ai_msg, "content") else ai_msg
                        st.markdown("### Answer (Direct LLM prompt)")
                        st.write(answer)
            except Exception as e:
                st.error(f"Error while querying model: {e}")

    # Show memory (optional)
    if enable_memory and memory is not None:
        st.markdown("### Conversation memory (in-session)")
        mem_text = memory.buffer if hasattr(memory, "buffer") else None
        if mem_text:
            st.write(mem_text)
        else:
            st.write("No memory recorded yet. (Memory works if used with chains that accept memory.)")

else:
    st.info("Upload a CSV to get started.")

# Footer: quick help
st.markdown("---")
st.markdown(
    """
**Notes / Tips**
- Set `GROQ_API_KEY` in your environment (or GitHub Secrets for deployment).  
- If you want RAG with embeddings, set `OPENAI_API_KEY` and enable 'Enable RAG'.  
- If `create_csv_agent` isn't available in your LangChain version, install `langchain-experimental` or update LangChain.  
- This app uses `langchain_groq.ChatGroq` as the Groq wrapper (see Groq LangChain docs).
"""
)