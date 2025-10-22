import os
from dotenv import load_dotenv
import streamlit as st

# **FINAL FIX**: Use the correct experimental package path
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq

# ==============================================================================
# OPTIMIZATION: CACHE AGENT CREATION
# ==============================================================================
@st.cache_resource
def get_agent(llm_model, uploaded_file):
    """Creates and returns the LangChain CSV Agent."""
    # We use a context manager on the file to ensure proper handling by the LangChain tool
    return create_csv_agent(
        llm_model, 
        uploaded_file, # Pass the Streamlit UploadedFile object
        verbose=True
    )

def main():
    load_dotenv()

    # Check for the GROQ API Key
    if os.getenv("GROQ_API_KEY") is None or os.getenv("GROQ_API_KEY") == "":
        st.error("GROQ_API_KEY is not set. Please set it as a Streamlit Secret or in your .env file.")
        return
    
    st.set_page_config(page_title="Ask your CSV (Powered by GROQ)")
    st.header("Ask your CSV (GROQ Edition) 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if csv_file is not None:
        
        # Initialize the GROQ LLM (Using a fast Llama model)
        llm = ChatGroq(
            temperature=0, 
            model_name="llama3-8b-8192" 
        )
        
        # Use the cached function to create the agent executor
        # This will handle the error from the Groq API due to the robust structure
        agent_executor = get_agent(llm, csv_file)
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input logic
        user_question = st.chat_input("Ask a question about your CSV:")

        if user_question:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner(text="Talking to GROQ..."):
                    try:
                        # Invoke the agent executor
                        # The input to invoke is just the string for this type of agent
                        response = agent_executor.invoke(user_question)
                        output_text = response["output"]
                        st.markdown(output_text)
                        
                        # Save assistant response to history
                        st.session_state.messages.append({"role": "assistant", "content": output_text})
                        
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I ran into an error: {e}"})

if __name__ == "__main__":
        main()
