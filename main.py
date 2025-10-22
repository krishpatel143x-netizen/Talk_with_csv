import os
from dotenv import load_dotenv
import streamlit as st

# 1. NEW IMPORT: Use the correct LangChain Groq integration
from langchain_groq import ChatGroq

# 2. NEW IMPORT: The CSV Agent has moved to the experimental package
# This resolves previous import errors
from langchain_agent_toolkits import create_agent


# Use Streamlit's cache to only create the agent once for better performance
@st.cache_resource
def get_agent_executor(llm_model, uploaded_file):
    """Creates and returns the LangChain CSV AgentExecutor."""
    # The function name is updated from the deprecated 'create_agent' 
    # to the correct 'create_csv_agent' for CSV files.
    return create_csv_agent(
        llm_model, 
        uploaded_file, 
        verbose=True
    )

def main():
    load_dotenv()

    # 3. API KEY CHECK: Check for the GROQ API Key
    if os.getenv("GROQ_API_KEY") is None or os.getenv("GROQ_API_KEY") == "":
        st.error("GROQ_API_KEY is not set. Please set it as a Streamlit Secret or in your .env file.")
        return
    
    st.set_page_config(page_title="Ask your CSV (Powered by GROQ)")
    st.header("Ask your CSV (GROQ Edition) 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if csv_file is not None:
        
        # 4. LLM INSTANTIATION: Use ChatGroq instead of OpenAI
        llm = ChatGroq(
            temperature=0, 
            # Llama 3 8B is a fast and capable Groq model for this task
            model_name="llama3-8b-8192" 
        )
        
        # Create the agent executor
        agent_executor = get_agent_executor(llm, csv_file)
        
        # Initialize chat history for a better user experience
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input logic (using the modern st.chat_input)
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
                        response = agent_executor.invoke({"input": user_question})
                        output_text = response["output"]
                        st.markdown(output_text)
                        
                        # Save assistant response to history
                        st.session_state.messages.append({"role": "assistant", "content": output_text})
                        
                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
        main()
