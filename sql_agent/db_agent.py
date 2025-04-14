from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_ollama.llms import OllamaLLM
from langchain.chains import create_sql_query_chain
from langchain_groq import ChatGroq



db = SQLDatabase.from_uri("sqlite:///chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM artists LIMIT 10;")


#load llm
# llm = OllamaLLM(model='llama3.2:1b')

llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-qwen-32b")

# chain = create_sql_query_chain(llm,db)

# response = chain.invoke({"question":"How many employees are there?"})


toolkit = SQLDatabaseToolkit(db=db, llm=llm)



agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

response = agent_executor.invoke({"input":"List the total sales per country. Which country's customers spent the most?"})

print(response['output'])

# db.run(response)

# print(db.run(response.split("SQLQuery:")[1].strip()))
