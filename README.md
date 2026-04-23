# 📄 AI Policy & Compliance Assistant

![App Preview](images/app-home.png)

An AI-powered internal policy assistant that uses a local RAG system to analyze policy documents, answer questions, generate summaries, and create training quizzes.

---

## 🚀 Features
- Upload policy PDFs or TXT files
- Grounded Q&A using document context only
- Automatic policy summaries
- AI-generated training quizzes
- Persistent vector database (ChromaDB)
- Clean Streamlit web interface

---

## 🧠 How it works
1. Upload policy documents (PDF or TXT)
2. Text is extracted and chunked
3. Data is embedded and stored in ChromaDB
4. Queries are answered using retrieved context only
5. Outputs include answers, summaries, and quizzes

---

## 🛠 Tech Stack
- Python
- LlamaIndex
- Ollama (qwen2.5)
- ChromaDB
- HuggingFace Embeddings
- Streamlit

---

## 📸 Demo

### Upload & Index Policies
![Indexing](images/indexed-success.png)

### Policy Q&A (with citations)
![Q&A](images/qa-output.png)

### Generated Summary
![Summary](images/summary-output.png)

### Training Quiz
![Quiz](images/quiz-output.png)

---

## ▶️ How to Run

### Start Here
Make sure the project folder is on your Desktop:

AI-Policy-Assistant

---

### Setup & Run

1. Open Terminal and go to the project folder
```bash
cd ~/Desktop/AI-Policy-Assistant
