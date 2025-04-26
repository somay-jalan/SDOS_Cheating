import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import concurrent.futures
import hashlib
import shutil # Import shutil for directory removal

st.set_page_config(page_title="Multi-PDF Chatbot Optimized", layout="centered")

# Template for answering with context
template_with_context = """
Use the following pieces of retrieved context to answer the question. If you don't know the answer based on the context, say so.
Context: {context}
Question: {question}
Answer:
"""

# Template for answering without context (fallback)
template_without_context = """
Answer the following question based on your general knowledge. If you don't know the answer, just say that you don't know.
Question: {question}
Answer:
"""

pdfs_directory = './pdfs/'
os.makedirs(pdfs_directory, exist_ok=True)
faiss_index_dir = './faiss_index'
processed_files_log = os.path.join(faiss_index_dir, 'processed_files.log')

@st.cache_resource
def get_embeddings():
    # Initialize Ollama embeddings
    return OllamaEmbeddings(model="nomic-embed-text")

def load_processed_files():
    # Load the set of processed file hashes from the log file
    if os.path.exists(processed_files_log):
        try:
            with open(processed_files_log, 'r') as f:
                return set(line.strip() for line in f)
        except Exception as e:
            st.warning(f"Could not read processed files log: {e}. Starting fresh.")
            return set()
    return set()

def save_processed_file(file_hash):
    # Append a file hash to the processed files log
    try:
        # Ensure directory exists before writing log
        os.makedirs(faiss_index_dir, exist_ok=True)
        with open(processed_files_log, 'a') as f:
            f.write(file_hash + '\n')
    except Exception as e:
        st.error(f"Error saving processed file hash {file_hash}: {e}")


@st.cache_resource
def get_vector_store(_embeddings):
    # Load FAISS index from disk if it exists, otherwise create a new one.
    if os.path.isdir(faiss_index_dir):
        index_file_path = os.path.join(faiss_index_dir, "index.faiss")
        if os.path.exists(index_file_path):
            try:
                print(f"Attempting to load FAISS index from: {faiss_index_dir}")
                return FAISS.load_local(faiss_index_dir, _embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.error(f"Error loading FAISS index from {faiss_index_dir}: {e}. Initializing a new one.")
                try:
                    shutil.rmtree(faiss_index_dir)
                    print(f"Removed potentially corrupted index directory: {faiss_index_dir}")
                except OSError as rm_error:
                    st.error(f"Could not remove corrupted directory {faiss_index_dir}: {rm_error}. Manual cleanup might be required.")
                    raise RuntimeError(f"Failed to clean up corrupted index directory: {faiss_index_dir}") from rm_error
        else:
             print(f"Directory {faiss_index_dir} exists, but index.faiss not found. Initializing new index.")

    print(f"Initializing new FAISS index in: {faiss_index_dir}")
    try:
        os.makedirs(faiss_index_dir, exist_ok=True)
        dummy_text = ["Initialize vector store"]
        vs = FAISS.from_texts(dummy_text, _embeddings)
        vs.save_local(faiss_index_dir)
        print(f"Successfully initialized and saved new FAISS index to {faiss_index_dir}")
        if os.path.exists(processed_files_log):
            try:
                os.remove(processed_files_log)
                print(f"Cleared processed files log: {processed_files_log}")
            except OSError as log_err:
                 st.warning(f"Could not remove processed files log {processed_files_log}: {log_err}")
        return vs
    except Exception as init_err:
        st.error(f"Failed to initialize FAISS index: {init_err}")
        raise RuntimeError("Critical error: Could not initialize FAISS vector store.") from init_err


# --- Get resources ---
embeddings = get_embeddings()
vector_store = get_vector_store(embeddings)
model = OllamaLLM(model="gemma3:1b", streaming=True)
processed_files = load_processed_files()

def get_file_hash(file_content):
    # Calculate MD5 hash for file content
    return hashlib.md5(file_content).hexdigest()

def process_pdf(file_content, file_name):
    # Load, chunk, and return documents from PDF content
    temp_file_path = None
    try:
        temp_file_path = os.path.join(pdfs_directory, f"temp_{hashlib.md5(file_content).hexdigest()}_{file_name}")
        with open(temp_file_path, "wb") as f:
            f.write(file_content)

        loader = PDFPlumberLoader(temp_file_path)
        documents = loader.load()

        if not documents:
            st.warning(f"No content extracted from {file_name}.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunked_documents = text_splitter.split_documents(documents)
        print(f"Processed {file_name}: Found {len(documents)} pages, split into {len(chunked_documents)} chunks.")
        return chunked_documents
    except Exception as e:
        st.warning(f"Could not process {file_name}: {e}")
        return []
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as rm_err:
                st.warning(f"Could not remove temporary file {temp_file_path}: {rm_err}")


def retrieve_docs(query, k=4):
    # Retrieve relevant documents using similarity search
    try:
        # Check if the vector store has any documents before searching
        # Note: FAISS doesn't have a direct count. A simple check is to see if retrieval works.
        # A more robust check might involve trying a dummy search or checking index properties if available.
        # For simplicity, we rely on the search potentially returning an empty list.
        if vector_store and hasattr(vector_store, 'index') and vector_store.index.ntotal > 0:
             return vector_store.similarity_search(query, k=k)
        else:
             print("Vector store is empty or not initialized properly. Skipping search.")
             return [] # Return empty list if store is empty
    except Exception as e:
        st.error(f"Error retrieving documents from FAISS: {e}")
        return []

def answer_question_stream(question, documents):
    # Generate an answer using the LLM.
    # If documents are provided, use context. Otherwise, use general knowledge.
    if documents:
        # Use context from retrieved documents
        print("Answering based on retrieved documents.")
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt_template = template_with_context
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | model
        input_dict = {"question": question, "context": context}
    else:
        # No documents found, try answering from general knowledge
        print("No relevant documents found. Answering from general knowledge.")
        prompt_template = template_without_context
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | model
        input_dict = {"question": question}

    try:
        # Stream the response from the language model
        for chunk in chain.stream(input_dict):
             token = chunk if isinstance(chunk, str) else getattr(chunk, "content", "")
             yield token
    except Exception as e:
        st.error(f"Error during language model inference: {e}")
        yield "Sorry, an error occurred while generating the answer."


# ========== Streamlit UI ==========
st.title("📚 PDF Q&A Assistant")

uploaded_files = st.file_uploader(
    "Upload one or more PDFs to add them to the knowledge base",
    type="pdf",
    accept_multiple_files=True,
    help="Upload PDF files. They will be processed and indexed for Q&A."
)

if uploaded_files:
    files_to_process_info = [] # Store (content, name, hash)
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.getvalue()
        file_hash = get_file_hash(file_content)
        if file_hash not in processed_files:
            files_to_process_info.append((file_content, uploaded_file.name, file_hash))
        else:
            st.info(f"'{uploaded_file.name}' (hash: {file_hash[:7]}...) has already been processed.")

    if files_to_process_info:
        all_new_chunks = []
        processed_hashes_in_batch = set()
        with st.spinner(f"Processing {len(files_to_process_info)} new PDF(s)... This may take a moment."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_file = {
                    executor.submit(process_pdf, content, name): (name, hash_val)
                    for content, name, hash_val in files_to_process_info
                }
                for future in concurrent.futures.as_completed(future_to_file):
                    file_name, file_hash = future_to_file[future]
                    try:
                        chunks = future.result()
                        if chunks:
                            all_new_chunks.extend(chunks)
                            processed_hashes_in_batch.add(file_hash)
                    except Exception as exc:
                        st.error(f'Error processing {file_name}: {exc}')

            if all_new_chunks:
                try:
                    with st.spinner("Adding new documents to the index..."):
                         # Ensure vector_store is ready before adding
                         if vector_store:
                             vector_store.add_documents(all_new_chunks)
                         else:
                              raise ValueError("Vector store is not initialized.")
                    with st.spinner("Saving updated index to disk..."):
                         if vector_store:
                            vector_store.save_local(faiss_index_dir)
                         else:
                             raise ValueError("Vector store is not initialized.")

                    for file_hash in processed_hashes_in_batch:
                         processed_files.add(file_hash)
                         save_processed_file(file_hash)

                    st.success(f"Successfully processed and indexed {len(processed_hashes_in_batch)} new PDF(s).")
                except Exception as e:
                    st.error(f"Error adding documents to FAISS or saving index: {e}")
            elif files_to_process_info:
                 st.warning("No new content could be extracted from the uploaded PDF(s) to add to the index.")


# --- Chat Interface ---
st.markdown("---")
st.header("Ask a Question")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about the content of the uploaded PDFs"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            with st.spinner("Thinking..."):
                # 1. Retrieve relevant documents (k=0 means no retrieval if you want to force general knowledge)
                # Set k > 0 to attempt retrieval first
                related_documents = retrieve_docs(question, k=4) # k=4 is default

                # 2. Generate answer using stream (handles empty documents internally)
                stream = answer_question_stream(question, related_documents)
                for chunk in stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

        except Exception as e:
             st.error(f"An error occurred: {e}")
             full_response = "Sorry, I encountered an error while processing your question."
             response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
