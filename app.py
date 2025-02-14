import streamlit as st
import openai
from pinecone import Pinecone
import pymongo
from pymongo import MongoClient
import tiktoken
import os
from dotenv import load_dotenv

# ---- LOAD ENVIRONMENT VARIABLES ----
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# ---- CONNECT TO PINECONE ----
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "txt"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=1536, 
        metric="cosine", 
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}  # ✅ Corrected Spec
    )

index = pc.Index(index_name)

# ---- CONNECT TO MONGODB ----
client = MongoClient(MONGO_URI)
db = client["txt_chatbot"]
collection = db["documents"]

# ---- FUNCTION TO CHUNK TEXT ----
def chunk_text(text, chunk_size=1000):  # ✅ Increased chunk size for better context
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = encoding.encode(text)
    return [encoding.decode(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

# ---- STREAMLIT UI ----
st.title("📄 Q&A Chatbot from TXT File")
uploaded_file = st.file_uploader("Upload a TXT file", type="txt")

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(text)

    st.success(f"✅ File processed into {len(chunks)} chunks.")

    for i, chunk in enumerate(chunks):
        # ✅ Check if chunk already exists in MongoDB
        if collection.find_one({"text": chunk}):
            continue  # Skip duplicate chunks

        # Store in MongoDB
        doc = {"chunk_id": i, "text": chunk}
        collection.insert_one(doc)

        # ✅ Use new OpenAI Embedding API format
        embedding = openai.embeddings.create(input=chunk, model="text-embedding-ada-002").data[0].embedding
        index.upsert([(str(i), embedding)])

    st.success("✅ Data stored successfully!")

# ---- Q&A CHAT ----
query = st.text_input("Ask a question:")
if query:
    query_embedding = openai.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding

    # ✅ Retrieve multiple chunks to improve answer quality
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    if results and results.get("matches"):
        # ✅ Fetch multiple chunks and merge them
        matched_chunks = []
        for match in results["matches"]:
            doc = collection.find_one({"chunk_id": int(match["id"])})
            if doc:
                matched_chunks.append(doc["text"])

        combined_context = " ".join(matched_chunks)  # Merge multiple chunks

        # ✅ Improve OpenAI prompt to force answering from the document
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a document-based Q&A chatbot. Only answer using the provided text. Do not make up information."},
                {"role": "user", "content": f"Context: {combined_context}\n\nQuestion: {query}\n\nAnswer strictly based on the above context."}
            ]
        )

        st.write("**Answer:**", response.choices[0].message.content)
    else:
        st.warning("No relevant data found.")
