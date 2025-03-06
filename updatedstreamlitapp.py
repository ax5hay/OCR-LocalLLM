import sys
import fitz
import json
import time
import requests
import streamlit as st
from loguru import logger
from datetime import datetime

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "deepseek:latest"

# Function to check if Ollama API is available
def check_api_health():
    try:
        logger.debug("Checking Ollama API health...")
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version = response.json().get('version', 'unknown')
            logger.success(f"Ollama API is healthy. Version: {version}")
            return True
        else:
            logger.error(f"Ollama API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("Connection error: Ollama API is not running or unreachable")
        return False
    except requests.exceptions.Timeout:
        logger.error("Timeout: Ollama API did not respond within the timeout period")
        return False
    except requests.exceptions.RequestException as e:
        logger.exception(f"Failed to connect to Ollama API: {str(e)}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error checking API health: {str(e)}")
        return False

# Function to get available models with error handling
def get_available_models():
    try:
        logger.debug("Fetching available Ollama models...")
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "models" not in data:
                logger.warning("API response missing 'models' key")
                return [DEFAULT_MODEL]
                
            models = [model["name"] for model in data["models"]]
            if not models:
                logger.warning("No models found in API response")
                return [DEFAULT_MODEL]
                
            logger.success(f"Found {len(models)} available models: {', '.join(models)}")
            return models
        else:
            logger.warning(f"Failed to fetch models list: HTTP {response.status_code}")
            return [DEFAULT_MODEL]
    except requests.exceptions.ConnectionError:
        logger.error("Connection error fetching models: Ollama API is not running or unreachable")
        return [DEFAULT_MODEL]
    except requests.exceptions.Timeout:
        logger.error("Timeout fetching models: API did not respond within the timeout period")
        return [DEFAULT_MODEL]
    except requests.exceptions.RequestException as e:
        logger.exception(f"Request error fetching models: {str(e)}")
        return [DEFAULT_MODEL]
    except json.JSONDecodeError as e:
        logger.exception(f"JSON parse error in models response: {str(e)}")
        return [DEFAULT_MODEL]
    except Exception as e:
        logger.exception(f"Unexpected error fetching models: {str(e)}")
        return [DEFAULT_MODEL]

# Function to run the model with enhanced logging and error handling
def run_model(prompt, model=DEFAULT_MODEL, temperature=0.25, top_k=40, top_p=0.9):
    start_time = time.time()
    logger.info(f"Request to model: {model} | T: {temperature} | K: {top_k} | P: {top_p}")
    logger.debug(f"Prompt length: {len(prompt)} chars")
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
    }
    
    try:
        logger.debug("Sending request to Ollama API...")
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60)
        
        if response.status_code != 200:
            error_msg = f"API returned error: HTTP {response.status_code}"
            logger.error(error_msg)
            try:
                error_detail = response.json()
                logger.error(f"Error details: {error_detail}")
            except:
                logger.debug("No JSON error details available")
            yield f"Error: API returned status code {response.status_code}"
            return
            
        logger.debug("Streaming response started")
        # Stream response
        full_response = ""
        chunk_count = 0
        last_log_time = time.time()
        last_progress_log = 0
        
        for chunk in response.iter_lines():
            if chunk:
                try:
                    data = chunk.decode("utf-8")
                    if '"response":"' in data:
                        text_chunk = json.loads(data)["response"]
                        full_response += text_chunk
                        chunk_count += 1
                        
                        # Log periodically (every 1 second) to avoid console spam
                        current_time = time.time()
                        if current_time - last_log_time > 1:
                            # Calculate tokens per second
                            elapsed = current_time - start_time
                            tokens_per_sec = chunk_count / elapsed if elapsed > 0 else 0
                            
                            # Log progress every ~500 tokens or if significant progress
                            if chunk_count - last_progress_log >= 500:
                                logger.info(f"Received {chunk_count} chunks ({tokens_per_sec:.1f} tokens/sec), response length: {len(full_response)} chars")
                                last_progress_log = chunk_count
                                
                            last_log_time = current_time
                            
                        yield text_chunk
                    elif '"done":true' in data:
                        elapsed_time = time.time() - start_time
                        tokens_per_sec = chunk_count / elapsed_time if elapsed_time > 0 else 0
                        logger.success(f"Response completed in {elapsed_time:.2f}s | Length: {len(full_response)} chars | {chunk_count} chunks | {tokens_per_sec:.1f} tokens/sec")
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error in chunk: {str(e)}")
                    logger.debug(f"Problematic chunk data: {data[:100]}...")
                    continue
                except Exception as e:
                    logger.exception(f"Error parsing chunk: {str(e)}")
                    continue
        
    except requests.exceptions.Timeout:
        error_msg = "Request timed out after 60 seconds"
        logger.error(error_msg)
        yield f"Error: {error_msg}"
    except requests.exceptions.ConnectionError:
        error_msg = "Connection error: Ollama API is not running or unreachable"
        logger.error(error_msg)
        yield f"Error: {error_msg}"
    except requests.exceptions.RequestException as e:
        logger.exception(f"Request error: {str(e)}")
        yield f"Error connecting to Ollama API: {str(e)}"
    except Exception as e:
        logger.exception(f"Unexpected error during model execution: {str(e)}")
        yield f"Unexpected error: {str(e)}"

# Function to extract text from a document with error handling
def extract_text_from_doc(uploaded_file):
    start_time = time.time()
    logger.info(f"Processing document: {uploaded_file.name} ({uploaded_file.type})")
    
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            try:
                # Read file to bytes
                file_bytes = uploaded_file.read()
                logger.debug(f"Read {len(file_bytes)} bytes from file")
                
                # Open PDF with PyMuPDF
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page_count = len(doc)
                logger.info(f"PDF has {page_count} pages")
                
                for i, page in enumerate(doc):
                    try:
                        page_text = page.get_text()
                        text += page_text
                        # Log progress at appropriate intervals
                        log_interval = max(1, min(5, page_count // 10))  # Adaptive logging
                        if i % log_interval == 0 or i == page_count - 1:
                            logger.debug(f"Processed page {i+1}/{page_count}")
                    except Exception as e:
                        logger.error(f"Error extracting text from page {i+1}: {str(e)}")
                        text += f"\n[Error extracting text from page {i+1}]\n"
                
                logger.debug(f"Successfully extracted text from all {page_count} pages")
            except fitz.FileDataError as e:
                logger.error(f"PDF parsing error: {str(e)}")
                return f"Error: Could not parse PDF file. The file may be corrupted or password-protected."
            except Exception as e:
                logger.exception(f"Error processing PDF: {str(e)}")
                return f"Error processing PDF: {str(e)}"
        else:
            try:
                file_bytes = uploaded_file.read()
                text = file_bytes.decode("utf-8")
                logger.info(f"Processed text file ({len(text)} chars)")
            except UnicodeDecodeError as e:
                logger.error(f"Text encoding error: {str(e)}")
                # Try alternative encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        logger.debug(f"Trying alternative encoding: {encoding}")
                        text = file_bytes.decode(encoding)
                        logger.success(f"Successfully decoded with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if not text:
                    return "Error: Could not decode text file with any supported encoding."
            except Exception as e:
                logger.exception(f"Error processing text file: {str(e)}")
                return f"Error processing text file: {str(e)}"
        
        elapsed_time = time.time() - start_time
        logger.success(f"Document extraction completed in {elapsed_time:.2f}s | Extracted {len(text)} chars")
        return text
    except Exception as e:
        logger.exception(f"Unexpected error extracting text: {str(e)}")
        return f"Error extracting text: {str(e)}"

# Cache function to improve performance
@st.cache_data(ttl=3600)  # Cache available models for 1 hour
def fetch_models():
    return get_available_models()

# ---------------- Streamlit App ----------------
def main():
    logger.info("Starting application")
    
    try:
        st.set_page_config(
            page_title="Chat with Local AI",
            page_icon="🧠",
            layout="wide"
        )
        
        st.title("🧠 Chat with Local AI (Ollama)")
        st.markdown("Chat with your local LLM, upload documents, and adjust hyperparameters.")

        # Check API health
        api_healthy = check_api_health()
        if not api_healthy:
            st.error("⚠️ Cannot connect to Ollama API. Please make sure Ollama is running on http://localhost:11434")
            logger.error("Ollama API health check failed, halting application execution")
            return

        # Sidebar for model settings
        st.sidebar.header("⚙️ Model Settings")
        
        # Model selection
        try:
            available_models = fetch_models()
            model_name = st.sidebar.selectbox("Select Model", available_models, index=0)
            logger.info(f"Selected model: {model_name}")
        except Exception as e:
            logger.exception(f"Error fetching or displaying models: {str(e)}")
            model_name = DEFAULT_MODEL
            st.sidebar.error(f"Error loading models. Using default: {DEFAULT_MODEL}")
        
        # Hyperparameters
        try:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05, 
                                      help="Higher values make output more random, lower values more deterministic")
            with col2:
                top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05, 
                                help="Nucleus sampling parameter")
            
            top_k = st.sidebar.slider("Top-K", 1, 100, 40, 5, 
                                    help="Limits vocabulary to top K tokens")
        except Exception as e:
            logger.exception(f"Error setting up hyperparameter controls: {str(e)}")
            temperature, top_p, top_k = 0.7, 0.9, 40  # Default values
            st.sidebar.error("Error setting up parameter controls. Using defaults.")
        
        # Log parameter changes
        if "prev_params" not in st.session_state:
            st.session_state.prev_params = {"model": model_name, "temperature": temperature, "top_k": top_k, "top_p": top_p}
        
        current_params = {"model": model_name, "temperature": temperature, "top_k": top_k, "top_p": top_p}
        if current_params != st.session_state.prev_params:
            changes = {k: current_params[k] for k in current_params if st.session_state.prev_params[k] != current_params[k]}
            logger.info(f"Parameter changes: {json.dumps(changes)}")
            st.session_state.prev_params = current_params
        
        # File upload section
        st.sidebar.header("📂 Upload Documents")
        
        try:
            uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"], 
                                                  help="Upload a document to provide context for your queries")
            
            document_text = ""
            if uploaded_file:
                # Use a status placeholder instead of spinner in sidebar
                status_placeholder = st.sidebar.empty()
                status_placeholder.info("Processing document...")
                
                try:
                    logger.info(f"Processing uploaded file: {uploaded_file.name} ({uploaded_file.type})")
                    document_text = extract_text_from_doc(uploaded_file)
                    
                    if document_text.startswith("Error:"):
                        status_placeholder.error(document_text)
                        logger.error(f"Document processing failed: {document_text}")
                    else:
                        doc_preview = document_text[:200] + "..." if len(document_text) > 200 else document_text
                        status_placeholder.success(f"Document loaded: {uploaded_file.name}")
                        with st.sidebar.expander("Document Preview"):
                            st.write(doc_preview)
                        logger.success(f"Document loaded successfully: {uploaded_file.name} ({len(document_text)} chars)")
                except Exception as e:
                    status_placeholder.error(f"Error processing document: {str(e)}")
                    logger.exception(f"Error processing document: {str(e)}")
                    document_text = ""
        except Exception as e:
            logger.exception(f"Error in file upload section: {str(e)}")
            st.sidebar.error(f"Error processing file upload: {str(e)}")
            document_text = ""

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            logger.debug("Initialized empty chat history")

        # Display chat history in main area
        st.subheader("💬 Chat History")
        
        try:
            chat_container = st.container()
            with chat_container:
                for i, (role, text) in enumerate(st.session_state.chat_history):
                    if role == "You":
                        st.info(f"**{role}:** {text}")
                    else:
                        st.success(f"**{role}:** {text}")
        except Exception as e:
            logger.exception(f"Error rendering chat history: {str(e)}")
            st.error("Error displaying chat history")
        
        # User input section with clear labels
        st.subheader("💬 Your Message")
        
        try:
            user_input = st.text_area("Enter your message", height=100, placeholder="Type your message here...")
            
            col1, col2 = st.columns([1, 5])
            
            with col1:
                send_button = st.button("Send 🚀", use_container_width=True)
            with col2:
                if st.button("Clear Chat 🗑️", use_container_width=False):
                    st.session_state.chat_history = []
                    logger.info("Chat history cleared")
                    st.experimental_rerun()  # Use experimental_rerun for older Streamlit versions
            
            if send_button:
                if user_input.strip():
                    logger.info(f"User input received ({len(user_input)} chars)")
                    
                    # Append document context if available
                    if document_text:
                        final_prompt = f"Document context:\n{document_text}\n\nUser query:\n{user_input}"
                        logger.info(f"Added document context ({len(document_text)} chars) to prompt")
                    else:
                        final_prompt = user_input
                    
                    # Display user message
                    st.session_state.chat_history.append(("You", user_input))
                    
                    # Create a placeholder for the response
                    with st.spinner("AI is thinking..."):
                        response_placeholder = st.empty()
                        
                        # Generate response
                        start_time = time.time()
                        full_response = ""
                        
                        try:
                            for chunk in run_model(final_prompt, model_name, temperature, top_k, top_p):
                                full_response += chunk
                                # Update the response in real-time
                                response_placeholder.markdown(f"**AI:** {full_response}▌")
                            
                            elapsed_time = time.time() - start_time
                            logger.success(f"Response generated in {elapsed_time:.2f}s ({len(full_response)} chars)")
                            
                            # Update chat history and finalize response
                            st.session_state.chat_history.append(("AI", full_response))
                            response_placeholder.empty()
                            
                            # rerun to refresh the chat display
                            st.experimental_rerun()  # Use experimental_rerun for older Streamlit versions
                        except Exception as e:
                            error_msg = str(e)
                            logger.exception(f"Error generating response: {error_msg}")
                            st.error(f"Error generating response: {error_msg}")
                else:
                    logger.warning("User attempted to send empty message")
                    st.warning("Please enter a message before sending.")
        except Exception as e:
            logger.exception(f"Error in user input section: {str(e)}")
            st.error(f"Application error: {str(e)}")
    
    except Exception as e:
        logger.exception(f"Critical application error: {str(e)}")
        st.error(f"The application encountered a critical error: {str(e)}")

if __name__ == "__main__":
    try:
        logger.info("Application startup")
        main()
        logger.info("Application executed successfully")
    except Exception as e:
        logger.critical(f"Fatal application error: {str(e)}")
        print(f"FATAL ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)