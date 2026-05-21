from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import os
import time
import re
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

# Balanced settings: speed + quality
MAX_CHUNK_CHARS = 3500
MAX_CHUNKS = 6


COMMON_RULES = """
You must create clean, exam-ready notes that will be rendered as Markdown and downloaded as a PDF.

GENERAL FORMAT:
- Use proper Markdown headings.
- Use this structure:
  ## UNIT 1 — Topic Name
  ### Q1. Question Name
- Keep language simple and exam-oriented.
- Do not write unnecessary long paragraphs.
- Use bullet points for concepts and definitions.
- Use tables only when information is truly tabular.
- Do not generate broken Markdown.

BOXES:
Use blockquotes for important boxes.

For notes:
> **Note:** Write the important exam tip here.

For exam questions:
> **Exam Question:** Write the question here.

For diagrams:
> **Diagram to draw:** Describe the diagram clearly with labels.

TABLE RULES:
- Use proper Markdown tables for comparisons, truth tables, formulas, operators, and laws.
- Do not convert tables into bullet points.
- Never write a table in one single line.
- Every table row must be on a separate line.
- Every table row must start with | and end with |.
- Always keep a blank line before and after a table.
- Do not use display equations inside table cells.
- Do not put long formulas inside table cells.
- Inside table cells, use only short inline math like $\\hat{x} = x$.
- If the formula is long, write "See formula below" in the table cell and then write the full equation below the table.
- Do not split one table row across multiple lines.
- Do not leave half table rows like "| Momentum |" without all columns.

Correct table format:

| Quantity | Operator | Meaning |
|---|---|---|
| Position | $\\hat{x} = x$ | Measures position |
| Momentum | See formula below | Measures momentum |
| Energy | See formula below | Measures total energy |

Then write long formulas separately:

$$
\\hat{p} = -i\\hbar \\frac{\\partial}{\\partial x}
$$

$$
\\hat{E} = i\\hbar \\frac{\\partial}{\\partial t}
$$

EQUATION RULES:
- Use KaTeX-compatible LaTeX.
- Use $...$ only for small inline expressions.
- Use $$...$$ for important or long equations.
- Do not use \\[ \\].
- Do not use \\( \\).
- Do not write lone $ symbols.
- Do not write equations as plain broken text.
- Do not put display equations inside tables.
- Put every important equation on its own display block.

Correct equation examples:

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

FINAL CHECK BEFORE ANSWERING:
- No lone $ symbols.
- No one-line tables.
- No broken table rows.
- No long equations inside tables.
- No raw LaTeX outside $...$ or $$...$$.
"""


def create_chunks_from_pdf(file_path):
    """
    Reads PDF page by page and creates chunks without storing full PDF text.
    This prevents Render memory crashes.
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
                if current_chunk.strip():
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

Create a clean exam-ready summary from the PDF content.

Include:
- Unit-wise headings
- Important concepts
- Definitions
- Formulas
- Proper Markdown tables where required
- Notes
- Exam questions
- Diagram-to-draw boxes

Very important:
- Keep tables proper.
- Do not put long formulas inside table cells.
- Put long formulas below the table.
- Use $$...$$ for important equations.
- Do not create broken Markdown.

{COMMON_RULES}

PDF Content:
{text}
"""

    elif output_type == "Important Questions":
        return f"""
You are a college exam question paper expert.

Generate important exam questions from the PDF.

Divide into:
1. 2-mark questions
2. 5-mark questions
3. 10-mark questions
4. Numericals if present
5. Viva questions

Rules:
- Use proper headings.
- Use question boxes for important questions.
- Use proper tables for comparison-based questions.
- Write formulas cleanly using LaTeX.
- Do not put long equations inside tables.

{COMMON_RULES}

PDF Content:
{text}
"""

    elif output_type == "MCQs":
        return f"""
Create 20 MCQs from the PDF content.

Rules:
- Give 4 options for each question.
- Mark the correct answer.
- Add a short explanation.
- Use simple exam-level language.
- If formulas are present, write them using proper LaTeX.
- Avoid very long answers.

{COMMON_RULES}

PDF Content:
{text}
"""

    elif output_type == "Formula Sheet":
        return f"""
Create a clean formula sheet from the PDF content.

Rules:
- Extract all important formulas.
- Use proper formula headings.
- Write every important formula in display format using $$...$$.
- Explain symbols below each formula.
- Add where each formula is used.
- Use compact tables only when readable.
- Do not put long equations inside table cells.
- Add note boxes for common exam mistakes.

Format:

## Formula Name

$$
formula here
$$

Where:
- $symbol$ = meaning

Used for: explanation.

> **Note:** Important exam tip here.

{COMMON_RULES}

PDF Content:
{text}
"""

    elif output_type == "Viva Questions":
        return f"""
Create viva questions and answers from the PDF content.

Rules:
- Questions should be simple.
- Answers should be short.
- Focus on oral viva preparation.
- Use formulas only when required.
- Use note boxes for important viva tips.

{COMMON_RULES}

PDF Content:
{text}
"""

    else:
        return f"""
Summarize this PDF content in simple exam-ready language.

{COMMON_RULES}

PDF Content:
{text}
"""


def create_chunk_prompt(chunk_text, chunk_number, total_chunks):
    return f"""
You are reading part {chunk_number} of {total_chunks} from a college PDF.

Create a short, clean exam-oriented summary of this part only.

Extract:
- Important concepts
- Definitions
- Important formulas
- Important tables
- Notes
- Diagram-to-draw points
- Important exam questions

Very important:
- Do not create one-line tables.
- Do not put long equations in table cells.
- Write long equations below tables.
- Do not leave lone $ symbols.
- Use clean Markdown.

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""


def create_final_prompt(combined_notes, output_type):
    return f"""
You are an exam preparation assistant.

The notes below are partial summaries from different PDF parts.
Combine them into one final clean output.

Output type required: {output_type}

Final output rules:
- Remove repetition.
- Use proper unit-wise headings.
- Preserve important formulas.
- Preserve important tables.
- Format tables properly.
- Do not convert tables into bullet points.
- Do not put long equations inside tables.
- Put long equations below tables using $$...$$.
- Remove broken table rows.
- Remove lone $ symbols.
- Add note boxes, exam question boxes, and diagram boxes where useful.
- Make the answer look like a professional exam-preparation PDF.

Very strict table example:

Correct:

| Quantity | Operator | Meaning |
|---|---|---|
| Position | $\\hat{{x}} = x$ | Measures position |
| Momentum | See formula below | Measures momentum |

$$
\\hat{{p}} = -i\\hbar \\frac{{\\partial}}{{\\partial x}}
$$

Wrong:
| Quantity | Operator | Meaning | |---|---|---| | Momentum | $\\hat p = ...$ | Meaning |

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

Strict formatting rules:
- Output must be valid Markdown.
- Use Markdown tables correctly.
- Every Markdown table row must be on a new line.
- Never write a full table in one line.
- Never put display equations inside table cells.
- In table cells, use short inline math only.
- Put long formulas below tables using $$...$$.
- Use $...$ only for short inline math.
- Use $$...$$ for long or important equations.
- Never output lone $ symbols.
- Never use \\[ \\] or \\( \\).
- Never write raw broken LaTeX.
- Use blockquotes for notes, exam questions, and diagram boxes.
"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
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


def remove_orphan_dollars(text):
    """
    Removes lone $ symbols that appear on separate lines.
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        if line.strip() == "$":
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_math_delimiters(text):
    """
    Converts unsupported math delimiters into Markdown math.
    """
    text = text.replace("\\[", "$$")
    text = text.replace("\\]", "$$")
    text = text.replace("\\(", "$")
    text = text.replace("\\)", "$")
    return text


def repair_one_line_tables(text):
    """
    Repairs common AI mistake:
    | A | B | |---|---| | X | Y |
    into proper multi-line Markdown table.
    """
    lines = text.split("\n")
    repaired = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("|") and stripped.count("|") >= 8:
            possible_rows = re.split(r"\|\s+\|", stripped)

            fixed_rows = []
            for row in possible_rows:
                row = row.strip()

                if not row:
                    continue

                if not row.startswith("|"):
                    row = "| " + row

                if not row.endswith("|"):
                    row = row + " |"

                row = re.sub(r"\s*\|\s*", " | ", row)
                row = row.strip()

                fixed_rows.append(row)

            if len(fixed_rows) >= 2:
                repaired.append("")
                repaired.extend(fixed_rows)
                repaired.append("")
            else:
                repaired.append(line)
        else:
            repaired.append(line)

    return "\n".join(repaired)


def protect_tables_from_long_equations(text):
    """
    Prevents long equations inside Markdown table cells.
    If a table row contains long LaTeX, replace it with 'See formula below'
    and place the formula after the table.
    """
    lines = text.split("\n")
    output = []
    delayed_formulas = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2

        if is_table_line:
            in_table = True

            has_long_formula = any(
                token in line for token in [
                    "\\frac", "\\partial", "\\int", "\\sum", "\\nabla"
                ]
            )

            if has_long_formula and len(line) > 120:
                formulas = re.findall(r"\$([^$]+)\$", line)

                for formula in formulas:
                    if any(
                        token in formula for token in [
                            "\\frac", "\\partial", "\\int", "\\sum", "\\nabla"
                        ]
                    ):
                        delayed_formulas.append(formula)

                line = re.sub(r"\$[^$]+?\$", "See formula below", line)

            output.append(line)
        else:
            if in_table and delayed_formulas:
                output.append("")
                for formula in delayed_formulas:
                    output.append("$$")
                    output.append(formula.strip())
                    output.append("$$")
                    output.append("")
                delayed_formulas = []

            in_table = False
            output.append(line)

    if delayed_formulas:
        output.append("")
        for formula in delayed_formulas:
            output.append("$$")
            output.append(formula.strip())
            output.append("$$")
            output.append("")

    return "\n".join(output)


def normalize_markdown_tables(text):
    """
    Adds blank lines around Markdown tables for proper rendering.
    """
    lines = text.split("\n")
    output = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2

        if is_table_line and not in_table:
            if output and output[-1].strip() != "":
                output.append("")
            in_table = True

        if not is_table_line and in_table:
            if output and output[-1].strip() != "":
                output.append("")
            in_table = False

        output.append(line)

    return "\n".join(output)


def remove_broken_pipe_lines(text):
    """
    Removes badly broken leftover pipe lines that are not valid table rows.
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if stripped == "|":
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            cell_count = stripped.count("|") - 1
            if cell_count <= 1 and "---" not in stripped:
                continue

        cleaned.append(line)

    return "\n".join(cleaned)


def clean_ai_output(text):
    """
    Final cleanup before sending output to frontend.
    """
    text = normalize_math_delimiters(text)
    text = repair_one_line_tables(text)
    text = protect_tables_from_long_equations(text)
    text = normalize_markdown_tables(text)
    text = remove_orphan_dollars(text)
    text = remove_broken_pipe_lines(text)

    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


def process_pdf_chunks(chunks, output_type):
    if not chunks:
        raise Exception("No readable text found in this PDF")

    if len(chunks) == 1:
        prompt = create_prompt(chunks[0], output_type)
        output = generate_with_groq(prompt, max_tokens=2200)
        return clean_ai_output(output)

    partial_summaries = []

    for index, chunk in enumerate(chunks):
        chunk_prompt = create_chunk_prompt(chunk, index + 1, len(chunks))

        partial_summary = generate_with_groq(chunk_prompt, max_tokens=700)
        partial_summary = clean_ai_output(partial_summary)

        partial_summaries.append(f"## Part {index + 1}\n\n{partial_summary}")

        time.sleep(0.5)

    combined_notes = "\n\n".join(partial_summaries)
    combined_notes = combined_notes[:9000]

    final_prompt = create_final_prompt(combined_notes, output_type)

    final_output = generate_with_groq(final_prompt, max_tokens=2200)
    final_output = clean_ai_output(final_output)

    return final_output


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ExamEase AI backend is running with fixed f-string LaTeX syntax"
    })


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