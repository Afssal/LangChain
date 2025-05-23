from langchain.schema import HumanMessage
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

df = pd.read_csv('/home/afsal/Downloads/Langchain/csv_agent/HR.csv').fillna(0)

llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")


agent = create_pandas_dataframe_agent(llm=llm,df=df,verbose=True,allow_dangerous_code=True)


CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""

question = 'What was the average DailyRate'

response = agent.invoke(CSV_PROMPT_PREFIX+question+CSV_PROMPT_SUFFIX)

print(response['output'])