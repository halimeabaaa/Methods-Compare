from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from weaviate import connect_to_weaviate_cloud
from weaviate.auth import AuthApiKey
import os
from dotenv import load_dotenv
from langchain_weaviate import WeaviateVectorStore

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")


wcs_cluster_url = os.getenv("WEAVIATE_URL")
wcs_api_key = os.getenv("WEAVIATE_API_KEY")

auth_config = AuthApiKey(api_key=wcs_api_key)

client = connect_to_weaviate_cloud(
    cluster_url=wcs_cluster_url,
    auth_credentials=auth_config,
)

WEAVIATE_TEXT_KEY = "page_content" # Varsayılan olarak 'text', şemanıza göre değiştirin!

# 3 farklı koleksiyonun retriever'larını hazırla
retriever1 = WeaviateVectorStore(
    client=client,
    index_name="SLIDINGChunks",
    embedding=embedding_model,
    text_key=WEAVIATE_TEXT_KEY
).as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# Sorgu
query = "Tip ogrencilerinin meslekte iyi bir sekilde devam etmeleri neye baglidir?"

docs1 = retriever1.invoke(query)


all_docs = docs1


context = "\n\n".join([doc.page_content for doc in all_docs])

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a technical AI assistant. Your task is to generate a clear, concise, and technically accurate answer "
     "to the user's question based on the provided documents.\n\n"
     "You will receive a user question and a collection of documents. These documents are retrieved from three different vector databases "
     "(CRAGChunk, RAGCunks, SLIDINGChunks), each representing a different retrieval and chunking strategy.\n\n"
     "Follow these steps when generating your answer:\n"
     "1. First, analyze the user's question.\n"
     "2. If the documents do not contain sufficient information to answer the question:\n"
     "   - Reformulate the question to make it more specific and aligned with the content of the documents.\n"
     "   - Use the revised question as the basis to generate your answer based on the documents.\n"
     "3. If the information is sufficient, answer the original question directly using the documents.\n\n"
     "**Important Rules:**\n"
     "- Rely strictly on the information found in the documents.\n"
     "- Do not generate or hallucinate any unsupported facts.\n"
     "- Provide brief, technical, and well-structured responses.\n\n"
     "Documents:\n{context}"),
    ("human", "{query}")
])

formatted_prompt = prompt.format(context=context, query=query)
response = llm.invoke(formatted_prompt)


print("\nCEVAP:\n", response.content)
client.close()