import os
from dotenv import load_dotenv
import streamlit as st

# 1. Groq and LangChain Imports
# Use the correct import path for the Groq model
from langchain_groq import ChatGroq
# The CSV agent is now in the experimental package
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType


# Use Streamlit's cache decorator to ensure the agent is only created once,
# which speeds up the application significantly.
@st.cache_resource
def get_agent_executor(llm_model, uploaded_file):
    """Creates and returns the LangChain CSV AgentExecutor using Groq."""
    
    # create_csv_agent handles reading the CSV into a pandas DataFrame and setting up the agent.
    # We use AgentType.ZERO_SHOT_REACT_DESCRIPTION as it's a robust default for Groq.
    return create_csv_agent(
        llm_model, 
        uploaded_file, 
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

def main():
    # Load environment variables (for GROQ_API_KEY)
    load_dotenv()

    # --- Setup and API Key Check ---
    st.set_page_config(page_title="Ask your CSV (Groq Edition)", layout="centered")
    
    st.title("Ask your CSV")
    st.markdown("📈 Powered by **Groq**'s high-speed LLMs for lightning-fast data analysis.")

    # Check for the GROQ API Key
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Please set it in your environment or a `.env` file.")
        return
    
    # --- File Upload ---
    csv_file = st.file_uploader("Upload a CSV file to begin analysis", type="csv")
    
    if csv_file is not None:
        
        # 1. Instantiate the Groq LLM
        llm = ChatGroq(
            temperature=0, 
            # Llama 3 8B is an excellent, fast model for tool/agent use cases
            model_name="llama3-8b-8192" 
        )
        
        # 2. Get the Agent Executor
        try:
            agent_executor = get_agent_executor(llm, csv_file)
        except Exception as e:
            st.error(f"Error creating agent: {e}")
            return
        
        # --- Chat Interface Setup ---
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # --- User Input and Agent Invocation ---
        user_question = st.chat_input("Ask a question about the data (e.g., 'What is the average age?', 'Show me the top 5 rows')")

        if user_question:
            # Display user message in chat
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner(text="Talking to Groq... Analyzing the data..."):
                    try:
                        # Invoke the agent executor
                        # The agent executes the necessary Python code against the DataFrame
                        response = agent_executor.invoke({"input": user_question})
                        output_text = response["output"]
                        st.markdown(output_text)
                        
                        # Save assistant response to history
                        st.session_state.messages.append({"role": "assistant", "content": output_text})
                        
                    except Exception as e:
                        # Display a user-friendly error
                        error_message = f"An error occurred during analysis. Please check your CSV format or try rephrasing the question. Details: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
        main()

