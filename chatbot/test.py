# import getpass
# import os
# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model
# from langchain_core.messages import HumanMessage
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import START, MessagesState, StateGraph
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# import streamlit as st
# import time

# load_dotenv()

# os.environ["LANGSMITH_TRACING"] = "true"

# if not os.environ.get("LANGSMITH_API_KEY"):
#     os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

# if not os.environ.get("GROQ_API_KEY"):
#     os.environ['GROQ_API_KEY'] = getpass.getpass("Enter your Groq api key")

# # Initialize the chat model
# model = init_chat_model("llama-3.2-1b-preview", model_provider="groq")

# # Initialize the workflow
# workflow = StateGraph(state_schema=MessagesState)

# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You talk like a Swag. Answer all questions to the best of your ability",
#         ),
#         MessagesPlaceholder(variable_name="messages")
#     ]
# )

# # Define the function to call the model
# def call_model(state: MessagesState):
#     prompt = prompt_template.invoke(state)
#     response = model.invoke(prompt)
#     return {"messages": response}

# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)

# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)

# config = {"configurable": {"thread_id": "abc123"}}

# def modify_response(response):
#     for word in response.split():
#         yield word + " "
#         time.sleep(0.1)

# # Initialize session state for storing messages if not present
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display all previous messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Handle user input
# if prom := st.chat_input("What is up"):
#     # Append the user message to session state
#     st.session_state.messages.append({"role": "user", "content": prom})

#     # Display the user message
#     with st.chat_message("user"):
#         st.markdown(prom)

#     # Prepare the input for the model
#     input_messages = [HumanMessage(prom)]

#     # Get the response from the model
#     output = app.invoke({"messages": input_messages}, config)
#     res = output["messages"][-1]

#     # Display the assistant's response with simulated typing effect
#     with st.chat_message("assistant"):
#         response = st.write_stream(modify_response(res.content))

#     # Append the assistant's response to session state
#     st.session_state.messages.append({"role": "assistant", "content": response})

#     # Ensure to display the updated conversation
