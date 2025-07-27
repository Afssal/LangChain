import chromadb
from embeddings import embed
import uuid

class vector:

    def __init__(self,model_name,path,docs,top_k):

        self.model = model_name
        self.path = path
        self.docs = docs
        # self.query = query
        self.top_k = top_k
        


    def chroma(self):

        client = chromadb.PersistentClient(path=self.path)

        embedModel = embed(self.model)

        embeddingModel = embedModel.GetModel()

        ids = [str(uuid.uuid4()) for _ in self.docs]

        self.collection = client.create_collection(
            name="collection1",
            embedding_function=embeddingModel,
        )


        self.collection.add(
            documents=self.docs,
            ids=ids
        )

    def retrieval(self,query):

        return self.collection.query(
            query_texts=[query],
            n_results=self.top_k
        )

