# Detecting Hallucination of LLM Outputs using RAG

This project aims to detect hallucinations in the outputs of large language models (LLMs) by leveraging a Retrieval-Augmented Generation (RAG) approach. It uses cosine similarity between the model's output and retrieved context to determine factual consistency.

---

## Features
- **Verifier Class**: Checks the factual consistency of model outputs against retrieved context using Sentence Transformers.
- **Cosine Similarity**: Measures the similarity between the model's output and the context.
- **Threshold-Based Verdict**: Classifies outputs as "Factual" or "Hallucinated" based on a similarity threshold.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sgsvuppu/Detecting-Hallucination-of-LLM-Outputs-using-RAG.git
   cd Detecting-Hallucination-of-LLM-Outputs-using-RAG
2. Create and activate a virtual environment:
   python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies:
pip install -r requirements.txt

Usage
Import the Verifier class:

from rag_core import Verifier
Initialize the Verifier:

verifier = Verifier(threshold=0.72)
Install the required Python packages using:

pip install -r [requirements.txt](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2FUsers%2Fsaisrivastav%2FDesktop%2Frag-hallucination-demo%2Frequirements.txt%22%2C%22path%22%3A%22%2FUsers%2Fsaisrivastav%2FDesktop%2Frag-hallucination-demo%2Frequirements.txt%22%2C%22scheme%22%3A%22file%22%7D%7D)
Project Structure
rag_core.py: Contains the Verifier class for detecting hallucinations.
requirements.txt: Lists the dependencies for the project.
README.md: Documentation for the project.

How It Works
Encoding: The Verifier class uses the sentence-transformers/all-MiniLM-L6-v2 model to encode both the model's output (answer) and the retrieved context (context) into dense vector representations.
Cosine Similarity: The encoded vectors are compared using cosine similarity to measure their closeness.
Threshold-Based Classification: If the similarity score is greater than or equal to the specified threshold (default: 0.72), the output is classified as "Factual." Otherwise, it is classified as "Hallucinated."
License


