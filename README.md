# Text-Embedding-
Text Embedding and Question Answering with OpenAI

This repository contains a Python project that uses OpenAI's API to generate embeddings for text files, store them in a CSV file, and create a simple question-answering system based on these embeddings. The project also includes text cleaning, tokenization, and context creation functions.
Features

Text Cleaning and Tokenization: Cleans and tokenizes text files.
Embedding Generation: Generates text embeddings using OpenAI's text-embedding-ada-002 model.
Context Creation: Finds the most similar context for a given question based on the embeddings.
Question Answering: Uses OpenAI's gpt-3.5-turbo model to answer questions based on the created context.

Requirements

Python 3.7+
Required Python packages:
openai
pandas
numpy
nltk
python-dotenv

Installation

Clone the repository:

    sh

    git clone https://github.com/yourusername/your-repository-name.git
    cd your-repository-name

Create a virtual environment:

    sh

    python3 -m venv myenv
    source myenv/bin/activate

Install the required packages:

    sh

    pip install -r requirements.txt

Set up your OpenAI API key:

    Create a .env file in the root directory of the project.
    Add your OpenAI API key to the .env file:

    plaintext

    OPENAI_API_KEY=your-api-key

Download NLTK data:

python

    import nltk
    nltk.download('punkt')

Usage

    Prepare your text files:
        Place your text files in a directory and set the path in the text_directory variable in the script.

Run the script:

    sh

    python create_presentation.py

Answer questions:

When prompted, enter your question.
The script will generate an answer based on the most relevant context from the text files.


Project Structure

create_presentation.py: Main script for generating embeddings, cleaning text, and answering questions.
requirements.txt: List of required Python packages.
.env: Environment variables file (not included in the repository, create manually).
.gitignore: Specifies which files and directories to ignore in the repository.



License

This project is licensed under the MIT License.
Acknowledgements

OpenAI for providing the API.
NLTK for text processing tools.
