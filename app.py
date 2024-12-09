import streamlit as st
import os
from dotenv import load_dotenv 
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from git import Repo


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


pc = Pinecone(api_key=PINECONE_API_KEY)


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

def clone_repository(repo_url):
    repo_name = repo_url.split("/")[-1]  
    repo_path = f"./{repo_name}"  
    if os.path.exists(repo_path):
        print(f"Directory {repo_path} already exists. Skipping clone.")
        return repo_path
    Repo.clone_from(repo_url, repo_path)
    return repo_path

path = clone_repository("https://github.com/CoderAgent/SecureAgent")

SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java', '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}
IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git', '__pycache__', '.next', '.vscode', 'vendor'}

def get_file_content(file_path, repo_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        rel_path = os.path.relpath(file_path, repo_path)
        return {"name": rel_path, "content": content}
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def get_main_files_content(repo_path: str):
    files_content = []
    try:
        for root, _, files in os.walk(repo_path):
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)
    except Exception as e:
        print(f"Error reading repository: {str(e)}")
    return files_content

file_content = get_main_files_content(path)


def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

text = "I am a programmer"
embeddings = get_huggingface_embeddings(text)


pinecone_index = pc.Index("codebase-rag")
vectorstore = PineconeVectorStore(index_name="codebase-rag", embedding=HuggingFaceEmbeddings())

documents = []
for file in file_content:
    doc = Document(
        page_content=f"{file['name']}\n{file['content']}",
        metadata={"source": file['name']}
    )
    documents.append(doc)

vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=HuggingFaceEmbeddings(),
    index_name="codebase-rag",
    namespace="https://github.com/CoderAgent/SecureAgent"
)

# RAG
def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace="https://github.com/CoderAgent/SecureAgent")
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    system_prompt = f"""You are a Senior Software Engineer, specializing in TypeScript.

    Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
    """
    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    return llm_response.choices[0].message.content

# UI
st.set_page_config(page_title="Codebase RAG Chatbot", layout="wide")

st.title("Codebase RAG Chatbot")


message_container = st.container()

user_input = st.text_input("How may I assist you:")

if user_input:

    message_container.write(f"**You:** {user_input}")
    
    bot_response = perform_rag(user_input)

    message_container.write(f"**chatbot:** {bot_response}")
