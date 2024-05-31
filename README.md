## TOS Tina - A chatbot for N4


TOS Tina is a chatbot implemented using a base LLM + RAG retrieval on N4 documents and implemented in N4 website
To provide an interactive question-answering system based on a large collection of documents, including PDFs, HTML, and XML files. The system leverages advanced NLP techniques such as document loading, text splitting, embeddings, vector stores, LLMs, cross-attention models, and NER enrichment to provide accurate and context-aware answers.

## Prerequisites

* Python 
* Natural Language Processing (NLP)
* Large Language Models (LLM) 
* Flask
* HTML, CSS, Javascript

## Dependencies

* langchain
* langchain_community
* torch
* accelerate
* sentence_transformers
* faiss-cpu
* tiktoken
* huggingface-hub
* pypdf
* spacy
* ctransformers
* Flask
* pyttsx3
* bs4
* Python 3.6 or higher
* beautifulsoup4
* xml.etree.ElementTree 
  
**Please refer to requirements.txt file**

## Setup Instructions

1. Download mistral model - https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q8_0.gguf
2. Download mpnet base v2 model - https://huggingface.co/sentence-transformers/all-mpnet-base-v2
3. Create a virtual environment
4. Install reuirements.txt through pip installer

### Features in TOS Tina

*Loads and processes a large collection of documents (PDFs, HTML, and XML) 
*Splits text into manageable chunks for better NLP processing Creates embeddings for the text chunks using a Hugging Face model
*Creates a vector store using FAISS for efficient document retrieval
*Implements a CTransformers-based LLM for generating context-aware responses 
*Applies cross-attention re-ranking to improve the relevance of retrieved documents 
*Enriches text chunks with named entities using spaCy 
*Provides a Flask-based web interface for asking questions and receiving answers
*Generates audio responses using the pyttsx3 library Prerequisites

* **Named Entity Recognition (NER):**
  Named Entity Recognition was applied on N4 documents while creating a vector store. `en_core_web_sm` an English language multi-task Convolutional Neural Network(CNN) trained on OntoNotes
  was implemeted for applying NER.  NER helps in identifying and classifying key entities (such as names of people, organizations, locations, dates, and other proper nouns) within a text.

* **Cross Attention Document Retrieval:**
  The documents retrieved through the RAG are not specific to the query asked my user. Hence cross attention document retrival computes attention scores between user query and documents retrieved.
  The documents are then ranked according to the attention scores before being passed to the LLM. This was implemented using `ms-marco-MiniLM-L-6-v2` cross encoder.

* **HTML XML data in vector stores:**
   To use the data present in XML , HTML documents, Tos Tina utilises parsers. BeautifulSoup from bs4 and ElementTree from xml.etree are used to obtain textual information from HTML and XML files

* **Chat Functionality:**
	 Enables users to engage in real-time text-based conversations.
	 Supports a wide range of topics and queries.
	 Utilizes natural language processing to understand and respond to user input.
 
* **Microphone Input:**
	 Allows users to interact with the system using voice commands.
	 Enables hands-free operation, particularly useful in situations where typing may not be feasible.
	 Employs speech recognition technology to accurately interpret spoken words.
	 Enhances accessibility for users with disabilities or those who prefer verbal communication.
 
* **Audio Listening:**
  	Enables the system to receive and process audio input.
	  Supports listening to user queries, feedback, or commands.
	  Utilizes audio processing techniques to filter and analyze incoming audio data.

* **Saving Chat History:**
  	Provides the option to save past conversations for future reference.
	  Enables users to review previous interactions and track the progression of conversations.
  
## How to Run Locally

1. python -m venv <env_name>
2. Set-ExecutionPolicy Unrestricted  ( powershell)  [OR]
3. powershell Set-ExecutionPolicy Unrestricted ( cmd)
4. .\<env_name>\Scripts\activate
5. (vir_Env) PS D:\AI\Hackathon\sample> pip install -r requirements.txt
6. Provide path for models accordingly inside code (Mistral, mpnet base v2)
7. !python -m spacy download en_core_web_sm - If facing an error while loading spacy model
8. for downloading mpnet in the terminal enter a) git lfs install b) !git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
9. for downloading crossencoder model use these commands in terminal a)git lfs install b)git clone
https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
10. python app.py to run the application

Set the SECRET_KEY in the app.config:

app.config['SECRET_KEY'] = "some_random" Usage 

Run the Flask application: python app.py 

Access the application in your web browser at http://localhost:5000/.

Ask a question related to the loaded documents in the text input field and click "Ask".

The application will generate an audio response, which you can listen to by clicking the "Listen" button.

File Structure app.py: 

The main Flask application file.
static/: Contains static files like CSS and JavaScript. 
templates/: Contains HTML templates for rendering web pages.
audio/: Directory for storing generated audio files.
documents/: Directory containing the documents (PDFs, HTML, and XML) for processing.

This project is licensed under the Navis License.




  
