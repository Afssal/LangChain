from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


model = OllamaLLM(model='gemma3:1b')

#prompt template
template = 'You are a helpful assistant who answers questions clearly and concisely into a style that is {style} text : {text}.'

#create prompt template
prompt_template = ChatPromptTemplate.from_template(template)

#get given prompt template
print(prompt_template.messages[0].prompt.template)

#response style
respond_style = 'Respond in a pirate slang.'

#user input
query = "I received a damaged item. What should I do?"

"""format prompt template:here format is used instead format_message 
    template because ollama model expect template as single string
    not list
"""
user_input = prompt_template.format(
    style=respond_style,
    text=query
)

response = model(user_input)

print(response)
