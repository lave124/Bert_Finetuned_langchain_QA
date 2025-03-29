from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# 1. Load your BERT QA model
model_path = r"C:\Users\lovis\Desktop\Recommendation_system\Bert_FineTuned\github_Bert\Model_files"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# 2. Create custom QA function
def bert_qa(question: str, context: str) -> str:
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer

# 3. Set up LangChain components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load documents
loader = TextLoader("sample_text_rag.txt")
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create vector store
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Create simpler prompt template (using PromptTemplate instead of ChatPromptTemplate)
template = """Answer the question based only on the following context:
{context}

Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)

# 5. Create RAG chain with direct data flow
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def full_qa_chain(inputs):
    # Get the context from retrieved documents
    context = format_docs(retriever.invoke(inputs["question"]))
    # Format the prompt
    formatted_prompt = prompt.invoke({
        "context": context,
        "question": inputs["question"]
    }).text
    # Get the answer from BERT
    answer = bert_qa(inputs["question"], context)
    return answer

# Create the final chain
rag_chain = RunnablePassthrough.assign(
    answer=lambda x: full_qa_chain(x)
)

# 6. Run queries
while True:
    query = input("\nAsk a question (type 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    
    try:
        result = rag_chain.invoke({"question": query})
        print(f"\nAnswer: {result['answer']}")
        
        # Show sources
        docs = retriever.invoke(query)
        print("\nSources:")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.page_content[:200]}...")
    except Exception as e:
        print(f"\nError processing your question: {str(e)}")