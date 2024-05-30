## TOS Tina - A chatbot for N4


TOS Tina is a chatbot implemented using a base LLM + RAG retrieval on N4 documents and implemented in N4 website

The chatbot was developed using Flask Framework, HTML, CSS, Javascript.

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
  
**Please refer to requirements.txt file**

## Setup Instructions

1. Download mistral model - https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q8_0.gguf
2. Download mpnet base v2 model - https://huggingface.co/sentence-transformers/all-mpnet-base-v2
3. Create a virtual environment
4. Install reuirements.txt through pip installer

### Features in TOS Tina

* **Named Entity Recognition (NER):**
  Named Entity Recognition was applied on N4 documents while creating a vector store. `en_core_web_sm` an English language multi-task Convolutional Neural Network(CNN) trained on OntoNotes
  was implemeted for applying NER.  NER helps in identifying and classifying key entities (such as names of people, organizations, locations, dates, and other proper nouns) within a text.

* **Cross Attention Document Retrieval:**
  The documents retrieved through the RAG are not specific to the query asked my user. Hence cross attention document retrival computes attention scores between user query and documents retrieved.
  The documents are then ranked according to the attention scores before being passed to the LLM. This was implemented using `ms-marco-MiniLM-L-6-v2` cross encoder.

* **HTML XML data in vector stores:**
   To use the data present in XML , HTML documents, Tos Tina utilises parsers. BeautifulSoup from bs4 and ElementTree from xml.etree are used to obtain textual information from HTML and XML files
  
## How to Run Locally

1. python -m venv <env_name>
2. Set-ExecutionPolicy Unrestricted  ( powershell)  [OR]
3. powershell Set-ExecutionPolicy Unrestricted ( cmd)
4. .\<env_name>\Scripts\activate
5. (vir_Env) PS D:\AI\Hackathon\sample> pip install -r requirements.txt
6. Provide path for models accordingly inside code (Mistral, mpnet base v2)
7. !python -m spacy download en_core_web_sm - If facing an error while loading spacy model
8. for downloading mpnet in the terminal enter a) git lfs install b) !git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
9. python app.py to run the application




  
