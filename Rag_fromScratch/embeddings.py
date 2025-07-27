from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)


class embed:

    def __init__(self,model_name):

        self.model_name = model_name


    def GetModel(self):

        return OllamaEmbeddingFunction(
            url='http://localhost:11434',
            model_name=self.model_name
        )
