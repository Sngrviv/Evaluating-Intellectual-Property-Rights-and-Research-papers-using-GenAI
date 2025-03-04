# Evaluating-Intellectual-Property-Rights-and-Research-papers-using-GenAI

```markdown
# Evaluating Intellectual Property Rights and Research Papers using GenAI

This project provides a system to evaluate and protect intellectual property rights in research papers and documents using Generative AI. It includes features for document registration, plagiarism detection, blockchain-based ownership tracking, and analytics visualization.

## Features
- Upload and register research papers/documents as intellectual property
- Check for plagiarism and similarity with existing documents
- Track document ownership using a simple blockchain implementation
- Analyze document content using AI (requires Ollama service)
- Visualize IP statistics and trends
- Store and retrieve documents with metadata

## Prerequisites
- Python 3.8+
- Ollama (optional, for AI analysis) running on `localhost:11434`
- Git (for cloning the repository)

## Requirements
The project requires the following Python packages:
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.2
streamlit>=1.15.0
PyPDF2>=3.0.0
plotly>=5.3.0
requests>=2.26.0
```

## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/Evaluating-Intellectual-Property-Rights-and-Research-papers-using-GenAI.git
cd Evaluating-Intellectual-Property-Rights-and-Research-papers-using-GenAI
```

2. **Create a Virtual Environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **(Optional) Set up Ollama for AI Analysis**
- Install Ollama: Follow instructions at [Ollama's official site](https://ollama.ai/)
- Start the Ollama service:
```bash
ollama serve
```
- Ensure it's running on `http://localhost:11434`

## Usage

1. **Run the Application**
```bash
streamlit run ip_protection.py
```

2. **Access the Web Interface**
- Open your browser and go to `http://localhost:8501`
- Use the sidebar to navigate between:
  - Upload Document: Register new documents
  - Check Plagiarism: Verify document originality
  - View Database: See registered documents
  - Analytics: View IP statistics

3. **Features in Action**
- **Upload Document**: Upload a PDF, specify owner and type, and register it
- **Check Plagiarism**: Upload a PDF to check against registered documents
- **View Database**: Browse all registered documents with details
- **Analytics**: See visualizations of document types and submission trends

## Project Structure
```
├── ip_protection.py     # Main application file
├── ip_blockchain.json   # Blockchain storage (created on first run)
├── ip_database.json     # Document database (created on first run)
├── ip_protection.log    # Log file (created on first run)
├── requirements.txt     # Dependency list
└── README.md            # This file
```

## Configuration
- **Logging**: Configured to write to `ip_protection.log`
- **Storage**: Documents and blockchain data stored as JSON files
- **Ollama**: Must be running locally for AI analysis features

## Troubleshooting
- **Ollama Connection Issues**: Ensure Ollama is running and port 11434 is accessible
- **PDF Processing Errors**: Verify PDFs are valid and readable
- **Dependency Issues**: Check Python version and package installations

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m "Add your feature"`)
4. Push to branch (`git push origin feature/your-feature`)
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with Streamlit, Scikit-learn, and Plotly
- AI analysis powered by Ollama
- Created as of March 04, 2025
```

### Instructions:
1. **Copy and Paste**: Copy the entire content above and paste it into your `README.md` file.
2. **Update the Repository URL**: Replace `https://github.com/your-username/Evaluating-Intellectual-Property-Rights-and-Research-papers-using-GenAI.git` with your actual GitHub repository URL.
3. **Filename**: I’ve assumed the main Python file is named `ip_protection.py`. If it’s different (e.g., `main.py`), update the `streamlit run ip_protection.py` command accordingly.
4. **Requirements.txt**: Ensure you create a `requirements.txt` file with the listed dependencies. You can do this by running `pip freeze > requirements.txt` after installing them, or manually create it with the versions specified.
5. **License**: If you don’t have a `LICENSE` file yet, either create one (e.g., MIT License) or remove the license section if not applicable.

This README is ready to use and provides a professional, comprehensive guide for your project! Let me know if you need further adjustments.
