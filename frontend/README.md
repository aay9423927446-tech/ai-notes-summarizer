# ExamEase AI

ExamEase AI is an AI-powered study assistant that helps students convert academic PDFs into exam-ready notes.

Students can upload a PDF and generate:
- Summaries
- Important questions
- MCQs
- Formula sheets
- Viva questions
- Downloadable PDF notes

## Features

- Upload academic PDFs
- Extract text from PDFs
- Generate AI-powered notes using Groq API
- Supports mathematical equations using LaTeX/KaTeX
- Download generated notes as PDF
- Modern responsive UI
- Reset and loading animation

## Tech Stack

### Frontend
- React.js
- Vite
- React Markdown
- KaTeX
- jsPDF
- html2canvas

### Backend
- Python
- Flask
- pdfplumber
- Groq API
- Flask-CORS
- python-dotenv

## Project Structure

```text
ai-notes-summarizer/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── .env
│   ├── uploads/
│   └── venv/
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── .gitignore
└── README.md