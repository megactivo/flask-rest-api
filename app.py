from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
google_key = os.getenv('GOOGLE_API_KEY')
pinecone_key = os.getenv('PINECONE_API_KEY')

if not openai_key or not google_key or not pinecone_key:
    raise ValueError("One or more environment variables are not set.")

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key=google_key)
clientOAI = OpenAI(api_key=openai_key)

# Initialize Pinecone
# pinecone = Pinecone(
#     api_key=google_key,
#     environment="ns_megactivo_contabilidad, ns_megactivo_nomina"  # Replace with your Pinecone environment
# )

pinecone = Pinecone(
    api_key=pinecone_key,
    environment="us-east-1"  # Replace with your Pinecone environment
)

# print(pinecone.describe_index(name="megactivo-index-1"))

# Access the index through the Pinecone instance
index = pinecone.Index(name="megactivo-index-1")

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
        module = data.get("module")

        # Assign requested_namespace based on the value of module
        if module.upper() == "CONTABILIDAD":
            requested_namespace = "ns_megactivo_contabilidad"
        elif module.upper() == "CLIENTES":
            requested_namespace = "ns_megactivo_fe"
        elif module.upper() == "FACTURACION_ELECTRONICA":
            requested_namespace = "ns_megactivo_fe"
        elif module.upper() == "PROVEEDORES":
            requested_namespace = "ns_megactivo_proveedores"
        elif module.upper() == "NOMINA_ELECTRONICA":
            requested_namespace = "ns_megactivo_nomina"
        elif module.upper() == "CONFIGURACION":
            requested_namespace = "ns_megactivo_configuracion"
        elif module.upper() == "INVENTARIOS":
            requested_namespace = "ns_megactivo_inventarios"
        elif module.upper() == "POS":
            requested_namespace = "ns_megactivo_pos"
        elif module.upper() == "INFORMACION_EXOGENA":
            requested_namespace = "ns_megactivo_informacionexogena"
        else:
            return jsonify({"error": "Invalid module value"}), 400

        # resu = client.models.embed_content(
        #     model="gemini-embedding-exp-03-07", contents=question,
        #     config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        # )
        # question_embedded = resu.embeddings[0].values
        # Reduce the embedding dimension to match the Pinecone index
        # question_embedded = question_embedded[:1536]  # Use the first 1536 values

        # Generate embeddings for the question
        resu = clientOAI.embeddings.create(
            input=question,
            model="text-embedding-ada-002"
        )
        question_embedded = resu.data[0].embedding
        
        # Query Pinecone for the top 10 results
        pinecone_results = index.query(
            vector=question_embedded,
            top_k=10,
            include_metadata=True,
            namespace=requested_namespace,  # Replace with your Pinecone namespace
        )

        # Extract relevant information from Pinecone results
        top_results = [result['metadata']['text'] for result in pinecone_results['matches']]

        # print("Top results from Pinecone...")
        # for i, result in enumerate(top_results):
        #     print(f"Result {i + 1}: {result}")

        # Append the top results to the question
        question += "\n\nBasate en este conocimiento para dar tu respuesta: " + "\n".join(top_results)

        # Generate a response using the LLM
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25", 
            contents=question
        )

        return jsonify({"answer": response.text}), 201
        # return jsonify({"questionandknowledge": question}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()