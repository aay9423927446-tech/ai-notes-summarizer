from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "llama-3.1-8b-instant"

MAX_CHUNK_CHARS = 3500
MAX_CHUNKS = 8


COMMON_RULES = """
Formatting rules:
- Use proper Markdown headings.
- Use bullet points where needed.
- Do not create Markdown tables using | | |.
- Do not put many formulas in one horizontal line.
- If content has a table, convert it into a clean bullet list or numbered list.
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
"""


def create_chunks_from_pdf(file_path):
    """
    Reads PDF page by page and creates chunks without storing full PDF text.
    This prevents Render memory crash.
    """
    chunks = []
    current_chunk = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""

            if not page_text.strip():
                continue

            if len(current_chunk) + len(page_text) < MAX_CHUNK_CHARS:
                current_chunk += "\n" + page_text
            else:
                chunks.append(current_chunk.strip())
                current_chunk = page_text

            if len(chunks) >= MAX_CHUNKS:
                break

    if current_chunk.strip() and len(chunks) < MAX_CHUNKS:
        chunks.append(current_chunk.strip())

    return chunks


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
- If the PDF contains a table, convert it into bullet points
- Never generate table format using | symbols
- Write one formula per line, not side by side
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


def create_chunk_prompt(chunk_text, chunk_number, total_chunks):
    return f"""
You are reading part {chunk_number} of {total_chunks} from a college PDF.

Create a short exam-oriented summary of this part only.

Rules:
- Extract important concepts
- Extract definitions
- Extract formulas
- Keep it concise
- Use bullet points
- Use proper LaTeX for equations
- Do not make long explanations

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""


def create_final_prompt(combined_notes, output_type):
    return f"""
You are an exam preparation assistant.

The following notes are partial summaries created from different parts of a PDF.

Now combine them into one final polished output.

Output type required: {output_type}

Rules:
- Remove repetition
- Organize properly with headings
- Keep it exam-oriented
- Use simple student-friendly language
- Preserve important formulas
- Use clean LaTeX equations
- Do not create Markdown tables with | symbols
- If table-like data exists, convert it into bullet points

{COMMON_RULES}

Partial notes:
{combined_notes}
"""


def generate_with_groq(prompt, max_tokens=1200, retries=1):
    for attempt in range(retries + 1):
        try:
            chat_completion = client.chat.completions.create(
                model=MODEL_NAME,
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
- Never create Markdown tables using | symbols.
"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=max_tokens
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            error_message = str(e)

            if "rate_limit_exceeded" in error_message or "tokens per minute" in error_message:
                if attempt < retries:
                    time.sleep(65)
                    continue

            raise e


def process_pdf_chunks(chunks, output_type):
    if not chunks:
        raise Exception("No readable text found in this PDF")

    # Small PDF: direct generation
    if len(chunks) == 1:
        prompt = create_prompt(chunks[0], output_type)
        return generate_with_groq(prompt, max_tokens=2200)

    # Large PDF: chunk processing
    partial_summaries = []

    for index, chunk in enumerate(chunks):
        chunk_prompt = create_chunk_prompt(chunk, index + 1, len(chunks))
        partial_summary = generate_with_groq(chunk_prompt, max_tokens=800)
        partial_summaries.append(f"## Part {index + 1}\n{partial_summary}")

        time.sleep(2)

    combined_notes = "\n\n".join(partial_summaries)
    combined_notes = combined_notes[:9000]

    final_prompt = create_final_prompt(combined_notes, output_type)
    final_output = generate_with_groq(final_prompt, max_tokens=2500)

    return final_output


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ExamEase AI backend is running with memory-safe large PDF support"})


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

    try:
        chunks = create_chunks_from_pdf(file_path)
        ai_output = process_pdf_chunks(chunks, output_type)

        return jsonify({
            "message": "AI notes generated successfully",
            "text": ai_output
        })

    except Exception as e:
        error_message = str(e)

        if "rate_limit_exceeded" in error_message or "tokens per minute" in error_message:
            return jsonify({
                "error": "The PDF is large and the free AI limit was reached. Please wait 1 minute and try again."
            }), 500

        if "Request too large" in error_message:
            return jsonify({
                "error": "This PDF is still too large for the current free AI limit. Try uploading unit-wise PDFs."
            }), 500

        if "memory" in error_message.lower():
            return jsonify({
                "error": "The PDF is too heavy for the free server memory. Try a smaller PDF."
            }), 500

        return jsonify({"error": error_message}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)