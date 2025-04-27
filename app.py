from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key="AIzaSyARW29vXwnRaaIIbIeiHi5S8Nx7hhiMsAo")

# Initialize Pinecone
pinecone = Pinecone(
    api_key="pcsk_665KeU_GkauQNwvM8hqqaKkJALJYXNQABaxkzLQQQCSebQ8jxLWkmvSrJaAW2D4gw4kziW",
    environment="ns_megactivo_contabilidad, ns_megactivo_nomina"  # Replace with your Pinecone environment
)

# Access the index
index = pinecone.index("megactivo-index-1-cxritoj.svc.aped-4627-b74a")

# get all friends
@app.route("/", methods=["GET"])
def home():
    result = [{"name": "juan pablo", "role": "web developer", "description": "super good"},
              {"name": "maria del pilar", "role": "dibujante arquitectonica", "description": "super good"}]
    return jsonify(result)

# say hello
@app.route("/hello", methods=["POST"])
def say_hello():
    try:
        data = request.json
        name = data.get("name")

        return jsonify({"hello": "hello " + name}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# create a completion
@app.route("/completion", methods=["POST"])
def create_completion():
    try:
        data = request.json
        question = data.get("question")

        # Generate embeddings for the question
        question_embedded = client.models.generate_embeddings(
            model="gemini-2.0-flash", contents=question
        ).embeddings

        # Query Pinecone for the top 10 results
        pinecone_results = index.query(
            vector=question_embedded,
            top_k=10,
            include_metadata=True
        )

        # Extract relevant information from Pinecone results
        top_results = [result['metadata']['text'] for result in pinecone_results['matches']]

        # Append the top results to the question
        question += "\n\nGround with this knowledge: " + "\n".join(top_results)

        # Generate a response using the LLM
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=question
        )

        return jsonify({"answer": response.text}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()