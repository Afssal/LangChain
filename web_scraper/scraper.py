from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
# from langchain_ollama.llms import OllamaLLM
from langchain.chains import create_extraction_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pprint
from langchain_groq import ChatGroq






def scraping(url,schema):


    #scrap website using asynchromiumloader(best for handle js rendering)
    loader = AsyncChromiumLoader(url)
    content = loader.load()

    #transform html content to text
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(content,tags_to_extract=['span'])


    print(docs_transformed[0])

    #split extracted text to chunks
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)


    extracted_content = extract(schema=schema,content=splits[0].page_content)
    pprint.pprint(extracted_content)
    return extracted_content



#load llm
# llm = OllamaLLM(model='llama3.2:1b')

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")


schema = {
    "properties":{
        "news_article_title":{"type":"string"},
        "news_article_summary":{"type":"string"},
    },
    "required":["news_article_title","news_article_summary"],
}

def extract(content:str,schema:dict):
    return create_extraction_chain(schema=schema,llm=llm).run(content)

url = ['https://www.thehindu.com/']
extracted_content = scraping(url,schema)