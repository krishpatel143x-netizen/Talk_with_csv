import os
from dotenv import load_dotenv
import streamlit as st

#try:
    #from langchain_experimental.agents.agent_toolkits import create_csv_agent
#except ImportError:
    #from langchain_community.agent_toolkits import create_csv_agent

from langchain_groq import ChatGroq

@st.cache_resource
def get_agent(llm_model, uploaded_file):
    return create_csv_agent(
        llm_model, 
        uploaded_file,
        verbose=True
    )

def main():
    load_dotenv()

    if os.getenv("GROQ_API_KEY") is None or os.getenv("GROQ_API_KEY") == "":
        st.error("GROQ_API_KEY is not set. Please set it as a Streamlit Secret or in your .env file.")
        return
    
    st.set_page_config(page_title="Ask your CSV (Powered by GROQ)")
    st.header("Ask your CSV (GROQ Edition) 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if csv_file is not None:
        
        llm = ChatGroq(
            temperature=0, 
            model_name="llama3-8b-8192" 
        )
        
        agent_executor = get_agent(llm, csv_file)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("Ask a question about your CSV:")

        if user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner(text="Talking to GROQ..."):
                    try:
                        response = agent_executor.invoke(user_question)
                        st.markdown(response["output"])
                        
                        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                        
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I ran into an error: {e}"})

if __name__ == "__main__":
        main()
