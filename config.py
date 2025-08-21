import os
from dotenv import load_dotenv

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chat_models import init_chat_model

load_dotenv()

embedding_model = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en")
collection_name = "vit-rag"

doc_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name=collection_name,
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API"],
)


# Remember --> I saved my gemini-flash key as `GOOGLE_API_KEY` in the .env file, so it is automatically reloaded when invoke `load_dotenv`.
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
)
