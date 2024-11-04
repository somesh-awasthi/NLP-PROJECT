
---

# Symptom-Based Medical Data Retrieval for Diagnosis Aid

## Objective

This project enables a user to input symptoms and retrieves relevant medical data to aid in diagnosis. By searching a knowledge base (embedded in a vector database), the system identifies and returns relevant information based on the user’s query. This supports more informed, data-driven medical insights.

## Key Features

1. **Document Partitioning and Summarization**  
   Extracts text and tables from PDFs for database storage and preprocessing.

2. **Text Processing**  
   Advanced tokenization, POS tagging, and dependency parsing for accurate understanding of medical texts.

3. **Data Retrieval and Embedding**  
   Embedding-based retrieval allows for quick search and matching to user queries.

## Project Components

### 1. Document Partitioning and Summarization

Using `partition_pdf` and `partition_doc()` from `unstructured.partition.pdf`, the project performs:

- **Text Extraction**: Extracts raw text and table data (simple OCR on tables if necessary).
- **Chunking**: Splits lengthy text into manageable parts for efficient processing. Tables and text are appended to respective lists, which are then merged for further preprocessing.

### 2. Text Processing Pipeline

After text extraction, the following steps are applied for optimized search accuracy:

- **Tokenization and POS Tagging**: Segments the text and identifies parts of speech for each token.
- **Dependency Parsing and Stopword Removal**: Filters tokens by their POS tags and dependencies, removing unimportant words.
- **Lemmatization**: Reduces words to their base form for consistency.
- **Chunking**: Uses `RecursiveCharacterTextSplitter()` to divide preprocessed text into smaller chunks for embedding.

Each chunk is stored as a `Document` object to facilitate further analysis.

### 3. Embedding and Database Storage

To enable efficient retrieval:

```python
db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=chroma_path)
```

- **Embedding**: Text chunks are embedded with `OpenAIEmbeddings()` and stored in a `Chroma` database, allowing fast similarity search based on user input.

### 4. Query-Based Retrieval and Response Generation

When a user submits a query:

1. The query is embedded and compared with the documents in the knowledge base.
2. **Threshold-based Matching**: Results with a similarity score above 0.7 are shortlisted.
3. **Top 4 Results**: The system selects the top 4 matches, which are then appended to a prompt for the foundation model.
4. **Final Output Generation**: The foundation model refines the response based on the retrieved documents, providing a clear and relevant result.

## How to Use

1. **Install Dependencies**  
   Clone the repository and install required packages.

   ```bash
   git clone https://github.com/somesh-awasthi/NLP-PROJECT.git
   cd NLP-PROJECT
   pip install -r requirements.txt
   ```

2. **Load Data**  
   Prepare and load the medical data PDFs into the knowledge base using the partitioning and processing functions.

3. **Run Retrieval System**  
   Use the main script to input symptoms and retrieve relevant medical data.

## Example Usage

```python
# Input a symptom to retrieve relevant data
user_query = "headache and nausea symptoms"
response = retrieve_medical_data(user_query)
print(response)
```

## Technologies Used

- **Python**: Primary language for data processing and retrieval.
- **Chroma**: Vector database for storing and retrieving embeddings.
- **OpenAI Embeddings**: Generates embeddings for text matching and similarity searches.
- **Natural Language Processing**: Tokenization, POS tagging, lemmatization, and dependency parsing.
  
## Future Improvements

- **Expand Knowledge Base**: Incorporate more medical sources for comprehensive coverage.
- **Optimize Thresholding**: Adjust retrieval thresholds for improved accuracy.
- **Advanced Summarization Models**: Use advanced summarization techniques for concise, relevant results.


---
<!-- 
NLP-PROJECT
SmTm_AlRt
This repository contains the codebase for the SmTm_AlRt project.

Introduction
In this project, we are building an early disease predictor through symptoms. For example, if a patient is feeling unwell and types their symptoms, we will attempt to predict the disease. This will help patients identify early symptoms of diseases.

Technologies Used
Python: Used as the primary programming language for backend development and data processing.
React: Utilized for frontend development to create a dynamic user interface.
Croma DB (tentative): Initially considered as the database solution. There's a potential plan to migrate to Pinecone.
JupyterLab: Integrated into the development environment for interactive Python development and data analysis.
Other Libraries:
NumPy: Used for numerical computing and array manipulation.
pandas: Employed for data manipulation and analysis.
scikit-learn: Utilized for machine learning tasks and predictive modeling.
Setup
1. Python Virtual Environment: Set up a Python virtual environment for managing project dependencies.
python -m venv env
2. Activate Virtual Environment: Activate the virtual environment.
venv\Scripts\activate
3. Install Dependencies: Install required Python dependencies using pip.
pip install numpy pandas scikit-learn jupyterlab
4. Install all the necessary requirements 
pip install -r requirements.txt
5. for embedding I'm using Hugging Face Embeddings
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
model_name="sentence-transformers/all-MiniLM-L6-v2" -->
