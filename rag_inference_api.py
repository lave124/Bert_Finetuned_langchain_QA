from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from functools import lru_cache
import time

app = Flask(__name__)

# Configuration (same as your original code)
MODEL_PATH = r"C:\Users\lovis\Desktop\Recommendation_system\Bert_FineTuned\github_Bert\Model_files"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DOCUMENT_PATH = "sample_text_rag.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
MAX_CONTEXT_LENGTH = 512

# Load and quantize the model
@lru_cache(maxsize=1)
def load_qa_model():
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
    # Apply dynamic quantization to linear layers
    # quantized_model = torch.quantization.quantize_dynamic(
    #     model,
    #     {torch.nn.Linear},
    #     dtype=torch.qint8
    # )
    return model

@lru_cache(maxsize=1)
def load_qa_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH)

# Initialize components
model = load_qa_model()
tokenizer = load_qa_tokenizer()
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Load and prepare documents (same as original)
loader = TextLoader(DOCUMENT_PATH)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
texts = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create prompt template (same as original)
template = """Based on the  provided context:
{context}

Carefully Analyze the context and responds to the user Question


Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)

# QA functions (same logic as original)
def bert_qa(question: str, context: str) -> str:
    inputs = tokenizer(
        question, 
        context, 
        return_tensors="pt", 
        truncation=True, 
        max_length=MAX_CONTEXT_LENGTH
    )
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.route('/ask', methods=['POST'])
def ask_question():
    start_time = time.time()
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Retrieve context
        docs = retriever.invoke(question)
        context = format_docs(docs)

        # print(context)
        
        # Get answer
        answer_start_time = time.time()
        answer = bert_qa(question, context)
        print(answer)
        inference_time = time.time() - answer_start_time
        
        # Prepare sources
        sources = [doc.page_content[:200] + "..." for doc in docs]
        
        return jsonify({
            "answer": answer,
            "sources": sources,
            "inference_time": round(inference_time, 4)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def health_check():
    return jsonify({
        "status": "ready",
        "model": "Quantized BERT QA",
        "embedding_model": EMBEDDING_MODEL
    })

if __name__ == '__main__':
    print("Flask API is running. Use POST /ask endpoint with JSON {'question': 'your question'}")
    app.run(host='0.0.0.0', port=8000, threaded=True)