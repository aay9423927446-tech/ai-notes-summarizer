from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


COMMON_RULES = """
Formatting rules:
- Use proper Markdown headings.
- Use bullet points where needed.
- Do not create Markdown tables using | | |.
- Do not put many formulas in one horizontal line.
- If content has a table, convert it into a clean bullet list or numbered list.
- For "Operators and Observables", write each observable separately like:
  - Position:
    $$
    \\hat{x} = x
    $$
  - Momentum:
    $$
    \\hat{p} = -i\\hbar \\frac{\\partial}{\\partial x}
    $$
  - Kinetic Energy:
    $$
    \\hat{T} = \\frac{\\hat{p}^2}{2m}
    $$
- Write all equations in clean KaTeX-compatible LaTeX.
- For inline equations, use $...$.
- For big equations, use only this format:

$$
equation here
$$

- Do not use \\[ \\].
- Do not use \\( \\).
- Do not start equations with [ or end with ].
- Do not write equations in Markdown tables.
- Do not write raw LaTeX as normal text.
- Put each major formula on a separate line.
- Keep the answer student-friendly and exam-oriented.

Correct equation format examples:

$$
F = ma
$$

$$
m\\frac{d^2x}{dt^2} = -\\frac{\\partial V}{\\partial x}
$$

$$
i\\hbar \\frac{\\partial \\Psi(x,t)}{\\partial t}
=
\\left[
-\\frac{\\hbar^2}{2m}\\frac{\\partial^2}{\\partial x^2}
+
V(x,t)
\\right]
\\Psi(x,t)
$$

$$
i\\hbar \\frac{\\partial \\Psi}{\\partial t}
=
\\left[
-\\frac{\\hbar^2}{2m}\\nabla^2
+
V
\\right]
\\Psi
$$

Where:
- $i = \\sqrt{-1}$
- $\\hbar = \\frac{h}{2\\pi}$
- $\\Psi$ is the wave function
- $V$ is potential energy
"""


def create_prompt(text, output_type):
    if output_type == "Summary":
        return f"""
You are an exam preparation assistant.

Summarize the following PDF content in simple student-friendly language.

Make the answer:
- Pointwise
- Easy to understand
- Useful for exam revision
- Include definitions, formulas, and important concepts
- Use proper headings
- Explain formulas clearly
- Do not copy long paragraphs directly

{COMMON_RULES}

Content:
{text}
"""

    elif output_type == "Important Questions":
        return f"""
You are a college exam question paper expert.

From the following study material, generate important exam questions.

Divide them into:
1. 2-mark questions
2. 5-mark questions
3. 10-mark questions
4. Numericals if present
5. Viva questions

For numerical/formula-based questions:
- Write formulas using proper LaTeX
- Put important formulas in display format using $$...$$

{COMMON_RULES}

Content:
{text}
"""

    elif output_type == "MCQs":
        return f"""
Create 20 multiple choice questions from the following content.

Rules:
- Give 4 options for each question
- Mark the correct answer
- Add short explanation
- Use simple exam-level language
- If formula-based MCQs are present, write equations using LaTeX math format

{COMMON_RULES}

Content:
{text}
"""

    elif output_type == "Formula Sheet":
        return f"""
Create a clean formula sheet from the following content.

Rules:
- Extract all important formulas
- Write every formula in display LaTeX format using $$...$$
- Explain symbols below each formula
- Add where each formula is used
- Keep it short and exam-oriented
- Do not put formulas inside Markdown tables
- If the PDF contains a table, convert it into bullet points.
- Never generate table format using | symbols.
- Write one formula per line, not side by side.
- Make equations look like textbook-style equations

Use this format:

## Formula Name

$$
formula here
$$

Where:
- $symbol$ = meaning

Used for: explanation.

{COMMON_RULES}

Content:
{text}
"""

    elif output_type == "Viva Questions":
        return f"""
Create viva questions and answers from the following content.

Rules:
- Make questions simple
- Give short answers
- Focus on exam and oral viva preparation
- Use student-friendly language
- If any answer contains a formula, write it using LaTeX math format

{COMMON_RULES}

Content:
{text}
"""

    else:
        return f"""
Summarize this content in simple student-friendly language.

{COMMON_RULES}

Content:
{text}
"""


def generate_with_groq(prompt):
    chat_completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": """
You are a helpful exam preparation assistant for engineering students.

Very important:
- Always write mathematical equations using KaTeX-compatible LaTeX.
- Use $...$ for inline equations.
- Use $$...$$ for display equations.
- Never use \\[ \\] or \\( \\).
- Never write equations as plain text.
- Never start equations with square brackets like [i\\hbar.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=3000
    )

    return chat_completion.choices[0].message.content


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ExamEase AI backend is running with Groq API"})


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400

    pdf_file = request.files["pdf"]
    output_type = request.form.get("outputType", "Summary")

    if pdf_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(file_path)

    extracted_text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

        if extracted_text.strip() == "":
            return jsonify({"error": "No readable text found in this PDF"}), 400

        extracted_text = extracted_text[:12000]

        prompt = create_prompt(extracted_text, output_type)
        ai_output = generate_with_groq(prompt)

        return jsonify({
            "message": "AI notes generated successfully",
            "text": ai_output
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)