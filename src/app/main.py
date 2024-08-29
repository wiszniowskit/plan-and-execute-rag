import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os 

from langchain_core.tools import Tool
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)


from langchain.agents import create_react_agent
from langchain import hub 
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler)
from langchain_openai import ChatOpenAI, OpenAI

from agents.rag_agent import RAGAgent

from langchain_core.tools import Tool

load_dotenv()




def generate_response(prompt_input, doc_path="/Users/tomasz/plan-and-execute-rag/docs/"):
    model = ChatOpenAI(model= "gpt-4o-mini", temperature=0.7, api_key=os.environ["OPENAI_API_KEY"])
    
    planner = load_chat_planner(model)
    rag_tool = RAGAgent(doc_path = doc_path).create_RAG_tool()
    tools = [rag_tool]
    
    
    executor = load_agent_executor(model, tools, verbose=True)
    
    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
    
    st_callback = StreamlitCallbackHandler(st.container())
    
    response = agent.invoke({"input": prompt_input}, {"callbacks": [st_callback]})
    return response["output"]


if prompt := st.chat_input("What's the Tesla's approach to Cybersecurity and  Data Privacy?", ):
    with st.chat_message("user"):
        st.markdown(prompt)


    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.write(response)
    