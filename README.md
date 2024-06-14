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
