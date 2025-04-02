import getpass
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState,StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_core.messages import HumanMessage
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import START, MessagesState, StateGraph
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# import streamlit as st
# import time


#extract api keys from .env file
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"

if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

if not os.environ.get("GROQ_API_KEY"):
    os.environ['GROQ_API_KEY'] = getpass.getpass("Enter your Groq api key")

# Initialize the chat model
model = init_chat_model("llama-3.2-1b-preview", model_provider="groq")

#initialize embedding vector
embeddings = OllamaEmbeddings(model='nomic-embed-text:latest')

#initialize vector database
vector_store = Chroma(
    collection_name="sample_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

#pdf loader
loader = PyPDFLoader('/home/afsal/Downloads/rag/isi_28.03_11.pdf')
pages = loader.load()

#split pdf into chunks using recursive splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
all_splits = text_splitter.split_documents(pages)


#store split into vector database
_ = vector_store.add_documents(documents=all_splits)


#initialize graph
graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query:str):

    """Rerieve information related to a query."""
    retrived_docs = vector_store.similarity_search(query,k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrived_docs
    )
    
    return serialized,retrived_docs


def query_or_respond(state:MessagesState):

    """
    Generate tool call for retrieval or respond.
    """
    llm_with_tools = model.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])

    return {"messages":[response]}


tools = ToolNode([retrieve])



def generate(state:MessagesState):

    """Generate answer."""

    recent_tool_messages = []
    for message in reversed(state["messages"]):

        if message.type == "tool":

            recent_tool_messages.append(message)
        else:
            break
    
    tool_messages = recent_tool_messages[::-1]


    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state['messages']
        if message.type in ("human","System")
        or (message.type == "ai" and not message.tool_calls)
    ]

    prompt = [SystemMessage(system_message_content)] + conversation_messages


    response = model.invoke(prompt)
    return {"messages":[response]}



graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


input_message = "what is mediapipe"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()