from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datasets import load_dataset
from PIL import Image
import base64
import os
import numpy as np
import warnings
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (for allowing Next.js frontend to access this API)

# Initialize environment, models, etc.
warnings.filterwarnings("ignore")

# === Load the fashion dataset ===
ds = load_dataset("jinaai/fashion-captions-de")

# Initialize ChromaDB for image search
chroma_client = chromadb.PersistentClient(path="./data/fashion.db")
image_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()
fashion_collection = chroma_client.get_or_create_collection(
    "fashion_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

# Initialize the GPT-4 vision model
vision_model = ChatOpenAI(model="gpt-4o", temperature=0.0)
parser = StrOutputParser()

# Define the image prompt
image_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a talented fashion designer and you have been asked to style an outfit for a special event. Provide the suggestions in a simple and structured format using bullet points. Each suggestion should be a single bullet point with a short description. Answer the user's question using the given image context with direct references to parts of the images provided."
            " Maintain a conversational tone, use markdown for emphasis."
        ),
        (
            "user",
            [
                {
                    "type": "text",
                    "text": "what are some good ideas to match the outfit {user_query}",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_1}",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_2}",
                },
            ],
        ),
    ]
)

# === Define the LangChain Chain: Combining prompt, vision model, and parser ===
vision_chain = image_prompt | vision_model | parser  # Create the vision chain


# Helper function to format inputs for the vision model
def format_prompt_inputs(data, user_query):
    inputs = {"user_query": user_query}
    image_path_1 = data["uris"][0][0]  # Local file path
    image_path_2 = data["uris"][0][1]  # Local file path

    # Encode the first image
    with open(image_path_1, "rb") as image_file:
        image_data_1 = image_file.read()
    inputs["image_data_1"] = base64.b64encode(image_data_1).decode("utf-8")

    # Encode the second image
    with open(image_path_2, "rb") as image_file:
        image_data_2 = image_file.read()
    inputs["image_data_2"] = base64.b64encode(image_data_2).decode("utf-8")

    return inputs


# Define API endpoint for querying fashion styles
@app.route("/query-fashion", methods=["POST"])
def query_fashion():
    data = request.json
    user_query = data.get("query")

    # Perform image search in ChromaDB
    results = fashion_collection.query(query_texts=[user_query], n_results=2, include=["uris", "distances"])

    # Convert local file paths to accessible URLs
    image_uris = []
    for uri in results["uris"][0]:
        filename = os.path.basename(uri)
        image_url = f"http://localhost:5000/images/{filename}"
        image_uris.append(image_url)

    # Format the inputs for the vision model
    prompt_inputs = format_prompt_inputs(results, user_query)
    response = vision_chain.invoke(prompt_inputs)  # Invoke the vision chain with the prompt inputs

    return jsonify({
        "response": response,
        "images": image_uris
    })

@app.route("/image-to-image-search", methods=["POST"])
def image_to_image_search_route():
    file = request.files.get('image')
    
    if not file:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Convert image to the format expected by the search function
        img = Image.open(file)
        img = img.convert('RGB')
        img = img.resize((224, 224))  # Adjust size as per model input requirements
        img_array = np.array(img)
        
        # Generate embedding
        embedding = embedding_function([img_array])
        
        # Perform the query in ChromaDB
        results = fashion_collection.query(query_embeddings=embedding, n_results=3, include=["uris", "distances"])
        
        if results and results["uris"] and len(results["uris"][0]) > 0:
            image_uris = [f"http://localhost:5000/images/{os.path.basename(uri)}" for uri in results["uris"][0]]
            return jsonify({"images": image_uris})
        else:
            return jsonify({"error": "No similar images found."}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve images from the 'dataset/fashion' directory
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('dataset/fashion', filename)

@app.route("/", methods=["GET"])
def home():
    return "<h1>Fashion Styling API</h1><p>Use the /query-fashion endpoint to query for fashion suggestions.</p>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
