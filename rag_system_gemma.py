import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

print("Libraries imported for rag_system_gemma.") 

KNOWLEDGE_BASE_DIR = "knowledge_base_raw"
FAISS_INDEX_PATH = "faiss_index_gemma_rag"
EMBEDDINGS_MODEL_NAME = "thenlper/gte-small"
OLLAMA_MODEL = "gemma3:4b" 
RETRIEVER_K = 3 
CHUNK_SIZE = 750
CHUNK_OVERLAP = 100


PROMPT_TEMPLATE_STR = """
Based *only* on the following context, answer the question.
If the answer is not found in the context, respond with "I don't know". Do not make up information.
**Provide only the most direct and concise answer to the question. For example, if the question asks for a name, provide only the name. If it asks for a year, provide only the year. Do not include extra phrases, repeat the question, or offer additional details unless absolutely necessary to answer the question directly.**

Context:
{context}

Question: {question}

Concise Answer:
"""

# Core RAG Pipeline Initialization Function
def get_gemma_rag_chain():
    """
    Initializes and returns the RAG (RetrievalQA) chain.
    This function handles embedding model loading, vector store preparation (load or create),
    LLM initialization, retriever creation, and RAG chain assembly.
    Returns the qa_chain object or None if any critical step fails.
    """
    print("--- Initializing RAG Pipeline (inside get_gemma_rag_chain function) ---")
    
    # 1. Initialize Embeddings Model
    print("Step 1: Initializing embeddings model...")
    start_time_embed = time.time()
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        print(f"  Embeddings model '{EMBEDDINGS_MODEL_NAME}' loaded in {time.time() - start_time_embed:.2f} seconds.")
    except Exception as e:
        print(f"  ERROR: Failed to initialize embeddings model: {e}")
        return None

    # 2. Prepare Vector Store
    print("\nStep 2: Preparing vector store...")
    start_time_vs = time.time()
    vectorstore = None

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"  Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        try:
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True 
            )
            print("  FAISS index loaded successfully.")
        except Exception as e:
            print(f"  ERROR loading FAISS index: {e}. Will attempt to recreate.")
            vectorstore = None 
    else:
        print(f"  No existing FAISS index found at {FAISS_INDEX_PATH}. Creating new one.")

    if vectorstore is None: 
        print(f"  Loading documents from {KNOWLEDGE_BASE_DIR} for new index creation...")
        try:
            loader = DirectoryLoader(
                KNOWLEDGE_BASE_DIR, 
                glob="**/*.txt", 
                loader_cls=TextLoader, 
                show_progress=True, 
                use_multithreading=True,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents = loader.load()
        except Exception as e:
            print(f"  ERROR: Failed to load documents: {e}")
            return None
            
        if not documents:
            print(f"  ERROR: No documents found in {KNOWLEDGE_BASE_DIR}. Cannot build RAG chain.")
            return None
        print(f"  Loaded {len(documents)} documents.")

        print("  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs_chunks = text_splitter.split_documents(documents)
        print(f"  Split into {len(docs_chunks)} chunks.")
        
        if not docs_chunks:
            print("  ERROR: No chunks created from documents. Cannot build RAG chain.")
            return None

        print("  Embedding document chunks and creating FAISS vector store (this may take a while)...")
        try:
            vectorstore = FAISS.from_documents(docs_chunks, embeddings)
            print("  FAISS vector store created.")
            print(f"  Saving FAISS index to {FAISS_INDEX_PATH}...")
            vectorstore.save_local(FAISS_INDEX_PATH)
            print("  FAISS index saved.")
        except Exception as e:
            print(f"  ERROR: Failed to create or save FAISS vector store: {e}")
            return None
            
    print(f"  Vector store preparation finished in {time.time() - start_time_vs:.2f} seconds.")
    
    if vectorstore is None:
        print("  CRITICAL ERROR: Vector store is not available after all attempts. Cannot build RAG chain.")
        return None

    # 3. Initialize LLM
    print("\nStep 3: Initializing LLM (via Ollama)...")
    print(f"  Ensure Ollama application is running with the '{OLLAMA_MODEL}' model pulled.")
    start_time_llm = time.time()
    try:
        llm = Ollama(model=OLLAMA_MODEL)
        print(f"  LLM '{OLLAMA_MODEL}' appears connected/initialized in {time.time() - start_time_llm:.2f} seconds.")
    except Exception as e:
        print(f"  ERROR: Failed to initialize Ollama LLM. Is Ollama running and '{OLLAMA_MODEL}' pulled? Error: {e}")
        return None

    # 4. Create Retriever
    print("\nStep 4: Creating retriever...")
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
        print(f"  Retriever created. Will retrieve top {retriever.search_kwargs['k']} documents.")
    except Exception as e:
        print(f"  ERROR: Failed to create retriever: {e}")
        return None

    # 5. Create RetrievalQA Chain
    print("\nStep 5: Creating RetrievalQA chain...")
    try:
        QA_PROMPT = PromptTemplate(
            template=PROMPT_TEMPLATE_STR, input_variables=["context", "question"]
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True, 
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        print("  RetrievalQA chain created.")
    except Exception as e:
        print(f"  ERROR: Failed to create RetrievalQA chain: {e}")
        return None
        
    print("--- RAG Pipeline Initialized Successfully (inside get_gemma_rag_chain function) ---")
    return qa_chain 

# Main execution block for testing this script directly
if __name__ == "__main__":
    print("--- Main Execution of rag_system_gemma.py: Testing RAG Chain Setup ---")
    
    my_qa_chain = get_gemma_rag_chain() 

    if my_qa_chain: 
        print("\n--- Testing RAG Chain with a Sample Question (from main script of rag_system_gemma.py) ---")
        sample_question = "Who is Pittsburgh named after?" 
        
        print(f"Sample Question: \"{sample_question}\"")
        
        invocation_start_time = time.time()
        try:
            # Invoke the chain with the question
            result = my_qa_chain.invoke({"query": sample_question})
            invocation_end_time = time.time()

            print(f"\nGenerated Answer (took {invocation_end_time - invocation_start_time:.2f} seconds):")
            print(result.get('result', "[No result field found]")) 

            print("\nSource Documents Retrieved:")
            source_documents = result.get('source_documents', [])
            if source_documents:
                for i, doc in enumerate(source_documents):
                    source_name = doc.metadata.get('source', 'Unknown source')
                    content_snippet = doc.page_content[:150].replace('\n', ' ') + "..." if len(doc.page_content) > 150 else doc.page_content.replace('\n', ' ')
                    print(f"  Source {i+1}: {source_name}")
            else:
                print("  No source documents were returned with the result.")

        except Exception as e:
            print(f"  ERROR during RAG chain invocation in main test: {e}")
            print("  This might be due to Ollama issues, model problems, or context window limits if k is too high.")
    else:
        print("Failed to initialize RAG chain in the main test of rag_system_gemma.py. Check logs above for errors.")

    print("\n--- Main Execution of rag_system_gemma.py Finished ---")