Prerequisites
You must have the following installed on your system:

Python 3.8+: Download from the official Python website.

Git: For cloning the repository.

A Stable Internet Connection: Required for the initial run to download the necessary models (like Qwen2 and sentence-transformers) from Hugging Face.

Step 1: Clone the Repository
First, retrieve the project files onto your computer.

Open your command line interface (CLI) or terminal.

Navigate to the directory where you want to save the project.

Run the following command to clone the repository (replace the bracketed text with your actual GitHub URL):git clone https://github.com/Rakeshvarma007/smart-study-buddy

Change into the new project directory:
cd smart-study-buddy

Step 2: Set Up a Virtual Environment
A virtual environment isolates your project's dependencies from other Python projects, preventing conflicts.

Create the virtual environment:python -m venv venv

Activate the virtual environment:

On macOS/Linux (or most Unix shells):source venv/bin/activate

On Windows (Command Prompt):venv\Scripts\activate

On Windows (PowerShell):.\venv\Scripts\Activate.ps1

Step 3: Install Dependencies
Next, you need to install all the necessary Python libraries.
Create a requirements.txt file in your project's root folder and add the following list of libraries:
streamlit
PyMuPDF
langchain-text-splitters
langchain-community
langchain-huggingface
transformers
torch
sentence-transformers

Install the required packages using: pip install -r requirements.txt

Step 4: Hugging Face Initialization (Recommended)
Logging into Hugging Face is recommended to prevent potential rate limits and ensure smooth model downloading.

Log in to Hugging Face:pip install huggingface-cli
huggingface-cli login

When prompted, paste your Hugging Face Token.

Step 5: Run the Smart Study Buddy
You are now ready to launch the application.

Run the Streamlit application: streamlit run smart_study_buddy.py

A new tab should automatically open in your web browser, pointing to the Streamlit app (usually at http://localhost:8501).

Step 6: Using the App
Upload your PDF: Use the "Upload Document" section in the left sidebar to upload a PDF file.

Wait for Indexing: The app will show initialization messages and a spinner while it processes the PDF, splits it into chunks, and creates the vector index. Wait for the "Index Ready! Ask a question." message in the sidebar.

Ask a Question: Use the chat input field at the bottom of the screen to ask a question about the content of your PDF.

Get Answers with Citations: The AI will answer your question using ONLY the text from your document and automatically cite the page numbers (e.g., [Page 5]).

When you are finished, stop the application by pressing Ctrl + C in your terminal, and exit your virtual environment with: deactivate.
