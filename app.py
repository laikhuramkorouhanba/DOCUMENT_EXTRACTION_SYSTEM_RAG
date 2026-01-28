from flask import Flask, request, render_template, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import login
import pandas as pd
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEndpoint




app = Flask(__name__)
login(token="Your_Hugging_Face_Api_Key")
callbacks = [StreamingStdOutCallbackHandler()]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B",
    max_new_tokens=250,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.2,
    repetition_penalty=1.2,
    callbacks=callbacks,
    streaming=True,
)


def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
csv_file_path = "/Users/laikhuramkorouhanbakhuman/Desktop/MAC/DOCUMENT_EXTRACTION/dataset/known_exploited_vulnerabilities.csv"
data = load_csv(csv_file_path)
def preprocess_data(data):
    data['combined'] = data[['cveID','vulnerabilityName', 'shortDescription', 'requiredAction', 'notes']].fillna('').agg(' '.join, axis=1)
    return data
data = preprocess_data(data)


# Retrieve relevant information using TF-IDF
def retrieve_information(query, data, top_n=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['combined'])
    query_vec = vectorizer.transform([query])

    # Calculate similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return data.iloc[top_indices], similarities[top_indices]

def augment_with_llama(query, retrieved_data):

    top_result = retrieved_data.iloc[0]['combined']
    prompt = (
        f"Query: {query}\n"
        f"Relevant Information: {top_result}\n\n"
        f"Response:\n"
    )
    response = llm.invoke(prompt)
    return response


@app.route('/')
def home():
    return render_template('index.html')  


@app.route("/query", methods=["POST"])
def query():
    print("Post /query called")
    json_content = request.form
    query = json_content.get("query")
    print(f"query: {query}")

    try:
        summary = llm.invoke(query)
        retrieved_data, similarities = retrieve_information(query, data)
        top_relevant_results = []
        for i, (index, similarity) in enumerate(zip(retrieved_data.index, similarities), start=1):
            result = {
                "vulnerabilityName": retrieved_data.loc[index, 'vulnerabilityName'],
                "similarity": f"{similarity:.2f}"
            }
            top_relevant_results.append(result)
        print(f"top relevant results: {top_relevant_results}" )
        response = augment_with_llama(query, retrieved_data)
        return jsonify({'summary': summary, 'response': response, 'top_relevant_results': top_relevant_results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
