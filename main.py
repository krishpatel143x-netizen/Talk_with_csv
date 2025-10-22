import os
from dotenv import load_dotenv
import streamlit as st

# 1. New Imports for GROQ
from langchain.agents import create_agent
from langchain_groq import ChatGroq

# ==============================================================================
# OPTIMIZATION: CACHE AGENT CREATION
# This function is essential to ensure the CSV file is only loaded and the agent
# is only created once when the file is uploaded, preventing memory issues.
# ==============================================================================
@st.cache_resource
def create_csv_agent(llm_model, csv_file):
    """Creates and returns the LangChain CSV Agent."""
    from langchain.agents import create_agent
    return create_agent(llm_model, csv_file)

def main():
    load_dotenv()

    # 2. Check for the GROQ API Key
    # Load the GROQ API key from the environment variable
    if os.getenv("GROQ_API_KEY") is None or os.getenv("GROQ_API_KEY") == "":
        st.error("GROQ_API_KEY is not set. Please set it in your environment variables or .env file.")
        print("GROQ_API_KEY is not set")
        exit(1)
    else:
        print("GROQ_API_KEY is set")

    st.set_page_config(page_title="Ask your CSV (Powered by GROQ)")
    st.header("Ask your CSV (GROQ Edition) 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if csv_file is not None:
        
        # 3. Initialize the GROQ LLM
        # We use ChatGroq, which is compatible with the LangChain Agent.
        # 'llama2-70b-4096' is a high-performance, free-tier model on the GROQ platform.
        llm = ChatGroq(
            temperature=0, 
            model_name="openai/gpt-oss-120b"
        )
        
        # Use the cached function to create the agent
        agent = create_csv_agent(llm, csv_file)
        agent.verbose = True
        

        user_question = st.text_input("Ask a question about your CSV: ")
   
        if user_question: # Streamlit treats an empty string as False
           with st.spinner(text="Talking to GROQ..."):
               # Invoke the agent with the user's question
               response = agent.invoke({"input": user_question})
               st.success("GROQ Response:")
               st.write(response)


if __name__ == "__main__":
        main()

