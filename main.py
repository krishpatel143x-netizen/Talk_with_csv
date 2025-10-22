from langchain_experimental.agents import create_csv_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import streamlit as st

# We are using the experimental library for create_csv_agent, 
# and ChatGroq for the high-speed Groq models.

def main():
    load_dotenv()

    # Load the Groq API key from the environment variable (replaces OpenAI key check)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key is None or groq_api_key == "":
        st.error("GROQ_API_KEY environment variable is not set. Please set it to run this application.")
        return
    else:
        # The print statement is kept from the original style for local debug visibility
        print("GROQ_API_KEY is set") 

    st.set_page_config(page_title="Ask your CSV")
    # Updated header to reflect the new LLM backend
    st.header("Ask your CSV 📈 (Powered by Groq)")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if csv_file is not None:
        try:
            # 1. Instantiate the Groq model
            # Use a fast Groq model like Mixtral 8x7b
            llm = ChatGroq(
                temperature=0, 
                model_name="mixtral-8x7b-32768"
            )

            # 2. Create the CSV Agent using the modern LangChain experimental package
            # We use 'openai-tools' as the agent_type for models that support function calling (like Groq's models).
            agent = create_csv_agent(
                llm,
                csv_file,
                verbose=True,
                # Setting agent_type is highly recommended for modern chat models
                agent_type="openai-tools" 
            )
            
            # The agent is now an AgentExecutor instance.

            user_question = st.text_input("Ask a question about your CSV: ")
    
            if user_question:
                with st.spinner(text="Groq is thinking..."):
                    # 3. Invoke the agent
                    # AgentExecutor's invoke returns a dictionary, so we access the 'output' key for the final answer.
                    response = agent.invoke({"input": user_question})
                    st.write(response["output"])
                    
        except Exception as e:
            st.error(f"An error occurred during agent execution: {e}")
            st.info("Please verify the CSV file is valid and your question is clear. Also, check the Groq API key.")


if __name__ == "__main__":
    main()

