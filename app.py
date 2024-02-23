from flask import Flask, render_template, request, jsonify
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import os
import google.generativeai as genai   

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Load environment variables
load_dotenv()
# Retrieve the Google API key from the environment variables
os.getenv("GOOGLE_API_KEY")
# Configure the Google Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_conversational_chain():
    """
    Create a conversational chain for question answering.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize ChatGoogleGenerativeAI model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    # Define conversation buffer memory
    buffer_memory = ConversationBufferMemory(memory_key="gemini_conversation")

    # Define a prompt template for user interactions
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        memory=buffer_memory  # Include the buffer memory
    )

    # Load the question-answering chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    """
    Process user input, perform similarity search, and generate a response using the conversational chain.
    """
    # Load embeddings for Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the FAISS vector store from the saved index
    new_db = FAISS.load_local("faiss_index", embeddings)
    # Perform similarity search based on user's question
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain for question answering
    chain = get_conversational_chain()
    
    # Generate a response using the chain
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST', 'GET'])
def chat():
    if request.method == 'POST':
        user_question = request.json.get('user_question')  # Use get method to safely access the 'user_question' key
        if user_question is not None:
            response = user_input(user_question)
            return jsonify({'response': response}), 200  # Return JSON response with status code 200
        else:
            return jsonify({'error': 'No user question provided'}), 400  # Return error message with status code 400 for bad request
    elif request.method == 'GET':
        return render_template('chat.html')  # Render the chat.html template for GET requests


if __name__ == '__main__':
    app.run(debug=True)
