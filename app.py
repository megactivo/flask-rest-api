from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key="AIzaSyARW29vXwnRaaIIbIeiHi5S8Nx7hhiMsAo")

# get all friends
@app.route("/",methods=["GET"])
def home():
    result = result = [{"name": "juan pablo", "role": "web developer", "description": "super good"},{"name": "maria del pilar", "role": "dibujante arquitectonica", "description": "super good"}]
    return jsonify(result)

# say hello
@app.route("/hello",methods=["POST"])
def say_hello():
    try:
        data = request.json
        name = data.get("name")

        return jsonify({"hello": "hello "+name}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# create a completion
@app.route("/completion",methods=["POST"])
def create_completion():
    try:
        data = request.json
        question = data.get("question")

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=question
        )

        return jsonify({"answer": response.text}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()