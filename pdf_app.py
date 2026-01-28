from flask import Flask, request, render_template, jsonify
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from huggingface_hub import login
import os
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEndpoint



app = Flask(__name__)
login(token="Your_Hugging_Face_Api_Key")
callbacks = [StreamingStdOutCallbackHandler()]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B",
    max_new_tokens=200,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.5,
    repetition_penalty=1.2,
    callbacks=callbacks,
    streaming=True,
)

folder_path = "db"
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)
raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)


@app.route('/')
def home():
    return render_template('index.html')  

@app.route("/query", methods=["POST"])


@app.route("/query", methods=["POST"])
def query():

    print("Post /query called")
    json_content = request.form
    query = json_content.get("query")
    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.05,
        },
    )

    document_chain = create_stuff_documents_chain(llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    print("Generating Response")
    result = chain.invoke({"input": query})

    return jsonify({'summary': result["answer"]})


@app.route("/pdf", methods=["POST"])
def pdf():
    file = request.files["file"]
    file_name = file.filename

    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    save_dir = os.path.join(cwd, "pdf")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_file = os.path.join(save_dir, file_name)
    file.save(save_file)
    print(f"filename: {file_name} saved to {save_file}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()


    try:
        return jsonify({'status': "Successfully Uploaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)




