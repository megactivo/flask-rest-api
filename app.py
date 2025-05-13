from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json

class Requerimiento(BaseModel):
    answer: str
    answer_rating: str
    request_rating: str
    tags: list[str]
    browsable: bool

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
google_key = os.getenv('GOOGLE_API_KEY')
pinecone_key = os.getenv('PINECONE_API_KEY')

if not openai_key or not google_key or not pinecone_key:
    raise ValueError("One or more environment variables are not set.")

app = Flask(__name__)
# CORS(app)
# CORS(app, resources={r"/*": {"origins": "https://aimegactivo.web.app/#/nomina"}}, methods=["POST", "GET", "OPTIONS"], allow_headers=["Content-Type"])
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["POST", "GET", "OPTIONS"], allow_headers=["Content-Type"])

# @app.after_request
# def add_cors_headers(response):
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type"
#     return response

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

        system_prompt = """
        "inquietud": Se te indicará indicar un texto con una inquietud de un usuario de la app megactivo.com

        "Contexto": Tambien se te indicarán 10 textos que son el contexto en que debes basarte para dar tu respuesta.

        Tu respuesta debe consistir en un único objeto JSON con los siguientes campos:  "answer" que es de tipo Text, los campos “answer_rating” y “request_rating” que son de tipo Number, el campo "tags" que es un list o array de Text, y el campo "browsable" que es de tipo Boolean.

        Campo "answer": Ofrece en una etiqueta <div> HTML formateada de forma moderna, una respuesta completa, detallada y útil a la inquietud, integrando toda la información pertinente y enlaces a videos si resultan relevantes. Utiliza un tono amigable, claro y orientado a brindar soluciones. Si no dispones de suficiente información para responder la inquietud, deja el campo "answer" vacío y asigna 0 al campo "answer_rating". 

        Campo "answer_rating": Califica en este campo del 0 al 10 qué tan completa y adecuada es la respuesta proporcionada en el campo "answer".

        Campo "request_rating": Califica en este campo del 0 al 10 el grado de claridad y pertinencia de la solicitud o inquietud del usuario.

        Campo "tags": Proporciona en este campo una lista de etiquetas o tags que permitan clasificar la inquietud (por ejemplo: "saldos", "cartera", "compras", "problemas de saldos", "problemas de envío a la DIAN", etc.).

        Campo "browsable": Indica true en este campo si la información proporcionada no es suficiente para responder completamente la consulta del usuario y crees que buscar en internet podría mejorar la respuesta; de lo contrario indica false.

        """

        # Generate a response using the LLM
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25", 
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=Requerimiento
            )
        )

        print("Request received 007")

        # Extract the "text" field from the response
        try:
            response_data = json.loads(response.model_dump_json())
            response_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            return jsonify({"error": "Failed to extract the 'text' field from the response", "details": str(e)}), 500

        print("Request received 008")

        # Return only the extracted "text" field
        return response_text, 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()