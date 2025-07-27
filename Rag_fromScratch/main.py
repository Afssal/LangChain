from dataload import pdfLoader
from textsplitter import splitter
from vectorstore import vector
import ollama


loader = pdfLoader('data\\story.pdf')

txt = loader.load()


doc_splitter = splitter(txt)

chunks = doc_splitter.parasplit()

# print(chunks)
vectordb = vector('nomic-embed-text:latest','db',chunks,top_k=5)
vectordb.chroma()

# print("\n".join(result['documents'][0]))
messages = []

def llm(chat):
    messages.append(
        {
            'role':'user',
            'content':chat
        }
    )
    stream = ollama.chat(model='llama3.2:latest',messages=messages,stream=True)
    response = ''
    for chunk in stream:
        part = chunk['message']['content']
        print(part,end='',flush=True)
        response = response+part
    
    messages.append(
        {
            'role':'assistant',
            'content':response
        }
    )

    print("")


while True:
    query = input("Enter your quesiton")
    if query.lower() != 'bye':

        result = vectordb.retrieval(query)
        
        context = "\n".join(result['documents'][0])
        prompt = f'''based on given context answer the question correctly. if you dont know the correct answer, just tell you dont know.
                    context : {context}
                    question : {query}
                    Answer : '''

        llm(prompt)
    else:
        break
