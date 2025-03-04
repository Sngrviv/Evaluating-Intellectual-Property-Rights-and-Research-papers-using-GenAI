import os
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import streamlit as st
import PyPDF2
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("ip_protection.log"), logging.StreamHandler()]
)
logger = logging.getLogger("ip_protection")

class IPDocument:
    """Class representing an intellectual property document with metadata."""
    
    def __init__(self, document_id: str, content: str, document_type: str, 
                 owner: str, created_at: datetime, metadata: Dict = None):
        self.document_id = document_id
        self.content = content
        self.document_type = document_type  # patent, trademark, copyright, trade secret
        self.owner = owner
        self.created_at = created_at
        self.metadata = metadata or {}
        self.hash = self._generate_hash()
        self.ai_analysis = {}  # Store AI analysis results
        
    def _generate_hash(self) -> str:
        """Generate a unique hash for the document content."""
        return hashlib.sha256(self.content.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert document to dictionary for storage."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "document_type": self.document_type,
            "owner": self.owner,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IPDocument':
        """Create document instance from dictionary."""
        return cls(
            document_id=data["document_id"],
            content=data["content"],
            document_type=data["document_type"],
            owner=data["owner"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data["metadata"]
        )


class IPBlockchain:
    """Simple blockchain implementation for IP document tracking."""
    
    def __init__(self, storage_path: str = "ip_blockchain.json"):
        self.chain = []
        self.storage_path = storage_path
        self._load_chain()
        
        # Create genesis block if chain is empty
        if not self.chain:
            self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the first block in the chain."""
        genesis_block = {
            "index": 0,
            "timestamp": datetime.now().isoformat(),
            "document_hashes": [],
            "previous_hash": "0",
            "hash": hashlib.sha256("genesis".encode()).hexdigest()
        }
        self.chain.append(genesis_block)
        self._save_chain()
        
    def add_document(self, document: IPDocument) -> bool:
        """Add a document to the blockchain."""
        # Get the last block
        last_block = self.chain[-1]
        
        # Create a new block
        new_block = {
            "index": last_block["index"] + 1,
            "timestamp": datetime.now().isoformat(),
            "document_hashes": last_block["document_hashes"] + [document.hash],
            "previous_hash": last_block["hash"],
            "document_id": document.document_id
        }
        
        # Calculate new hash
        block_string = json.dumps(new_block, sort_keys=True)
        new_block["hash"] = hashlib.sha256(block_string.encode()).hexdigest()
        
        # Add to chain and save
        self.chain.append(new_block)
        self._save_chain()
        return True
    
    def verify_document(self, document: IPDocument) -> bool:
        """Verify if document exists in blockchain."""
        for block in self.chain[1:]:  # Skip genesis block
            if document.hash in block["document_hashes"]:
                return True
        return False
    
    def _save_chain(self):
        """Save blockchain to disk."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.chain, f, indent=4)
    
    def _load_chain(self):
        """Load blockchain from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.chain = json.load(f)
            except Exception as e:
                logger.error(f"Error loading blockchain: {e}")
                self.chain = []


def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

class IPSimilarityDetector:
    """Detect similarity between IP documents."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.document_vectors = {}
    
    def add_document(self, document: IPDocument):
        """Add document to the similarity detector."""
        try:
            # Transform just this document
            vector = self.vectorizer.fit_transform([document.content])
            self.document_vectors[document.document_id] = vector
        except Exception as e:
            logger.error(f"Error adding document to similarity detector: {e}")
    
    def find_similar_documents(self, document: IPDocument, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find documents similar to the given document."""
        if not self.document_vectors:
            return []
        
        results = []
        try:
            # Transform the query document
            query_vector = self.vectorizer.transform([document.content])
            
            # Compare with all stored documents
            for doc_id, vector in self.document_vectors.items():
                similarity = cosine_similarity(query_vector, vector)[0][0]
                if similarity >= threshold:
                    results.append((doc_id, similarity))
            
            # Sort by similarity score (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
        
        return results
    
    def analyze_with_ollama(self, content: str, model: str = "llama2") -> Dict:
        """Analyze content using Ollama AI model."""
        if not check_ollama_connection():
            logger.error("Ollama service is not accessible")
            return {
                "error": "Ollama service is not accessible. Please ensure Ollama is running.",
                "originality_score": 0,
                "key_concepts": [],
                "potential_ip_aspects": []
            }
            
        try:
            # Truncate content if too long (Ollama typically has context limits)
            max_chars = 4000  # Adjust based on model's context window
            truncated_content = content[:max_chars] if len(content) > max_chars else content
            
            prompt = """Please analyze the following text for intellectual property aspects. 
Respond ONLY with a JSON object in the following format:
{
    "originality_score": <float between 0 and 1>,
    "key_concepts": [<list of main concepts found in the text>],
    "potential_ip_aspects": [<list of potential IP protection aspects>]
}

Text to analyze:

""" + truncated_content

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"  # Request JSON format
                },
                timeout=30
            )
            
            if response.status_code != 200:
                raise requests.exceptions.RequestException(f"Ollama API returned status code {response.status_code}")
                
            result = response.json()
            try:
                # Try to extract JSON from the response
                response_text = result.get("response", "")
                # Find JSON content between curly braces
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    analysis = json.loads(json_str)
                    
                    # Validate and sanitize the response
                    return {
                        "originality_score": float(analysis.get("originality_score", 0.5)),
                        "key_concepts": list(analysis.get("key_concepts", ["No concepts identified"])),
                        "potential_ip_aspects": list(analysis.get("potential_ip_aspects", ["No IP aspects identified"]))
                    }
                else:
                    raise json.JSONDecodeError("No JSON found in response", response_text, 0)
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error parsing Ollama response: {e}")
                # Create a structured response from unstructured text
                return {
                    "originality_score": 0.5,
                    "key_concepts": ["Response parsing error"],
                    "potential_ip_aspects": ["Unable to analyze"],
                    "raw_response": response_text
                }
                
        except requests.exceptions.Timeout:
            logger.error("Ollama API request timed out")
            return {
                "error": "Analysis timed out. Please try again.",
                "originality_score": 0,
                "key_concepts": [],
                "potential_ip_aspects": []
            }
        except Exception as e:
            logger.error(f"Error in Ollama analysis: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "originality_score": 0,
                "key_concepts": [],
                "potential_ip_aspects": []
            }


class IPProtectionDatabase:
    """Database for IP document storage and retrieval."""
    
    def __init__(self, storage_path: str = "ip_database.json"):
        self.storage_path = storage_path
        self.documents = {}
        self._load_documents()
    
    def add_document(self, document: IPDocument) -> bool:
        """Add document to database."""
        self.documents[document.document_id] = document.to_dict()
        self._save_documents()
        return True
    
    def get_document(self, document_id: str) -> Optional[IPDocument]:
        """Retrieve document by ID."""
        if document_id in self.documents:
            return IPDocument.from_dict(self.documents[document_id])
        return None
    
    def search_documents(self, query: Dict) -> List[IPDocument]:
        """Search documents based on query parameters."""
        results = []
        
        for doc_dict in self.documents.values():
            match = True
            for key, value in query.items():
                if key in doc_dict:
                    if key == "metadata":
                        # For metadata, check if the query items are a subset
                        for meta_key, meta_val in value.items():
                            if meta_key not in doc_dict["metadata"] or doc_dict["metadata"][meta_key] != meta_val:
                                match = False
                                break
                    elif doc_dict[key] != value:
                        match = False
            
            if match:
                results.append(IPDocument.from_dict(doc_dict))
        
        return results
    
    def _save_documents(self):
        """Save documents to disk."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.documents, f, indent=4)
    
    def _load_documents(self):
        """Load documents from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.documents = json.load(f)
            except Exception as e:
                logger.error(f"Error loading IP database: {e}")
                self.documents = {}


class IPRightsManager:
    """Main class for IP rights management."""
    
    def __init__(self):
        self.database = IPProtectionDatabase()
        self.blockchain = IPBlockchain()
        self.similarity_detector = IPSimilarityDetector()
    
    def register_ip(self, content: str, document_type: str, owner: str, 
                   metadata: Dict = None) -> Tuple[bool, str, Optional[str]]:
        """Register new intellectual property."""
        # Check for existing similar documents
        temp_doc = IPDocument(
            document_id="temp",
            content=content,
            document_type=document_type,
            owner=owner,
            created_at=datetime.now(),
            metadata=metadata
        )
        
        similar_docs = self.similarity_detector.find_similar_documents(temp_doc)
        
        # If similar documents exist, return information
        if similar_docs:
            return False, "Similar document already exists", similar_docs[0][0]
        
        # Create new document with UUID
        document_id = str(uuid.uuid4())
        new_doc = IPDocument(
            document_id=document_id,
            content=content,
            document_type=document_type,
            owner=owner,
            created_at=datetime.now(),
            metadata=metadata
        )
        
        # Add to database
        self.database.add_document(new_doc)
        
        # Record in blockchain
        self.blockchain.add_document(new_doc)
        
        # Add to similarity detector
        self.similarity_detector.add_document(new_doc)
        
        return True, "IP successfully registered", document_id
    
    def verify_ip_ownership(self, document_id: str, claimed_owner: str) -> bool:
        """Verify the ownership of an IP document."""
        document = self.database.get_document(document_id)
        if document and document.owner == claimed_owner:
            # Verify document exists in blockchain
            temp_doc = IPDocument(
                document_id=document.document_id,
                content=document.content,
                document_type=document.document_type,
                owner=document.owner,
                created_at=document.created_at,
                metadata=document.metadata
            )
            return self.blockchain.verify_document(temp_doc)
        return False
    
    def check_ip_infringement(self, content: str) -> List[Dict]:
        """Check if content infringes on any registered IP."""
        # Create temp document
        temp_doc = IPDocument(
            document_id="temp",
            content=content,
            document_type="check",
            owner="checker",
            created_at=datetime.now()
        )
        
        similar_docs = self.similarity_detector.find_similar_documents(temp_doc, threshold=0.6)
        
        results = []
        for doc_id, similarity in similar_docs:
            doc = self.database.get_document(doc_id)
            if doc:
                results.append({
                    "document_id": doc_id,
                    "similarity": similarity,
                    "document_type": doc.document_type,
                    "owner": doc.owner,
                    "created_at": doc.created_at.isoformat()
                })
        
        return results
    
    def process_pdf(self, pdf_file: BytesIO) -> Tuple[str, Dict]:
        """Process PDF file and extract content with analysis."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
            
            # Get AI analysis
            ai_analysis = self.similarity_detector.analyze_with_ollama(content)
            
            return content, ai_analysis
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return "", {}


def create_streamlit_ui():
    """Create Streamlit user interface."""
    st.set_page_config(page_title="IP Rights Protection System", layout="wide")
    
    st.title("IP Rights Protection System")
    
    # Check Ollama connection status
    ollama_status = check_ollama_connection()
    status_color = "green" if ollama_status else "red"
    status_text = "Connected" if ollama_status else "Not Connected"
    st.sidebar.markdown(f"### Ollama Status: :{status_color}[{status_text}]")
    
    if not ollama_status:
        st.sidebar.error("""
        Ollama is not accessible. Please ensure:
        1. Ollama is installed
        2. Ollama service is running
        3. Port 11434 is available
        """)
    
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload Document", "Check Plagiarism", "View Database", "Analytics"]
    )
    
    ip_manager = IPRightsManager()
    
    if page == "Upload Document":
        st.header("Upload Research Paper")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        owner_name = st.text_input("Document Owner")
        doc_type = st.selectbox("Document Type", ["research_paper", "patent", "article", "other"])
        
        if uploaded_file and owner_name and st.button("Process Document"):
            with st.spinner("Processing document... This may take a few moments."):
                content, ai_analysis = ip_manager.process_pdf(BytesIO(uploaded_file.read()))
                
                if content:
                    if "error" in ai_analysis:
                        st.error(ai_analysis["error"])
                    
                    success, msg, doc_id = ip_manager.register_ip(
                        content=content,
                        document_type=doc_type,
                        owner=owner_name,
                        metadata={"ai_analysis": ai_analysis}
                    )
                    
                    if success:
                        st.success(f"Document registered successfully! ID: {doc_id}")
                        
                        # Display analysis in a more structured way
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Document Analysis")
                            st.metric("Originality Score", 
                                    f"{ai_analysis.get('originality_score', 0)*100:.1f}%")
                            
                        with col2:
                            st.subheader("Key Concepts")
                            for concept in ai_analysis.get('key_concepts', []):
                                st.markdown(f"- {concept}")
                        
                        st.subheader("Potential IP Aspects")
                        for aspect in ai_analysis.get('potential_ip_aspects', []):
                            st.markdown(f"- {aspect}")
                    else:
                        st.error(msg)
                else:
                    st.error("Failed to process the PDF document. Please check if the file is valid.")
    
    elif page == "Check Plagiarism":
        st.header("Check for Plagiarism")
        check_file = st.file_uploader("Upload document to check", type="pdf")
        
        if check_file and st.button("Check Plagiarism"):
            with st.spinner("Analyzing document for potential similarities..."):
                content, _ = ip_manager.process_pdf(BytesIO(check_file.read()))
                if content:
                    infringements = ip_manager.check_ip_infringement(content)
                    
                    if infringements:
                        st.warning("Potential similarities found:")
                        for inf in infringements:
                            with st.expander(f"Similarity: {inf['similarity']:.1%}"):
                                st.write(f"Document Owner: {inf['owner']}")
                                st.write(f"Document Type: {inf['document_type']}")
                                st.write(f"Created: {inf['created_at']}")
                    else:
                        st.success("No significant similarities detected")
                else:
                    st.error("Failed to process the PDF document. Please check if the file is valid.")
    
    elif page == "View Database":
        st.header("Registered Documents")
        for doc_id, doc_data in ip_manager.database.documents.items():
            with st.expander(f"Document {doc_id}"):
                st.write(f"Owner: {doc_data['owner']}")
                st.write(f"Type: {doc_data['document_type']}")
                st.write(f"Created: {doc_data['created_at']}")
                if 'metadata' in doc_data and 'ai_analysis' in doc_data['metadata']:
                    st.json(doc_data['metadata']['ai_analysis'])
    
    elif page == "Analytics":
        st.header("IP Analytics")
        docs = ip_manager.database.documents
        
        # Document type distribution
        doc_types = pd.DataFrame([doc['document_type'] for doc in docs.values()])
        if not doc_types.empty:
            fig = px.pie(doc_types, names=0, title="Document Type Distribution")
            st.plotly_chart(fig)
        
        # Timeline of submissions
        dates = [datetime.fromisoformat(doc['created_at']) for doc in docs.values()]
        if dates:
            date_df = pd.DataFrame(dates, columns=['date'])
            fig = px.line(date_df.groupby(date_df['date'].dt.date).size().reset_index(),
                         x='date', y=0, title="Submissions Over Time")
            st.plotly_chart(fig)


if __name__ == "__main__":
    create_streamlit_ui()