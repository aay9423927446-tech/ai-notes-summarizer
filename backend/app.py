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

# Balanced settings: good speed + good quality
MAX_CHUNK_CHARS = 3500
MAX_CHUNKS = 6


COMMON_RULES = """
Formatting and design rules:
- Make the output look like a professional exam-preparation PDF.
- Use proper Markdown headings.
- Use clean exam-oriented formatting.
- Keep language simple and student-friendly.
- Use clear section headings like:
  ## UNIT 1 — Topic Name
  ### Q1. Question Name

Important visual style:
- Use headings clearly.
- Use proper tables wherever the content is tabular.
- Use note boxes for tips.
- Use exam-question boxes for important questions.
- Use diagram-to-draw boxes when a diagram is required.

Question boxes:
Write important questions exactly like this:

> **Exam Question:** Write the actual question here.

Note boxes:
Write important tips exactly like this:

> **Note:** Write the exam tip here.

Diagram boxes:
If a diagram is needed, write it exactly like this:

> **Diagram to draw:** Describe the labelled diagram clearly.
> Include labels, current/voltage directions, components, and what the student should draw.

Tables:
- Use proper Markdown tables whenever the content is comparison-based, truth-table-based, or tabular.
- Do not convert tables into bullet points.
- Do not write tables in one single line.
- Every table row must be on a new line.
- Always leave one blank line before and after a table.
- Use tables for:
  - Truth tables
  - Comparison tables
  - Formula summary tables
  - Advantages vs disadvantages
  - Operator/observable tables
- Keep table content short and readable.
- Do not put many formulas in one long table row.
- Use this exact Markdown table format:

| Column 1 | Column 2 | Column 3 |
|---|---|---|
| Row 1 data | Row 1 data | Row 1 data |
| Row 2 data | Row 2 data | Row 2 data |

Good table example:

| Quantity | Operator | Meaning |
|---|---|---|
| Position | $\\hat{x} = x$ | Measures position |
| Momentum | $\\hat{p} = -i\\hbar \\frac{\\partial}{\\partial x}$ | Measures momentum |
| Energy | $\\hat{E} = i\\hbar \\frac{\\partial}{\\partial t}$ | Measures total energy |

Equations:
- Write all equations in clean KaTeX-compatible LaTeX.
- For inline equations, use $...$ only for very small expressions.
- For important equations, always use display format.
- For display equations, use only this format:

$$
equation here
$$

- Do not use \\[ \\].
- Do not use \\( \\).
- Do not start equations with [ or end with ].
- Do not write raw formulas as plain text.
- Do not write equations in one broken line.
- Put each major formula on a separate line.
- Keep formulas clean and readable.
- Never write important equations inside table cells if they are too long.
- If an equation is long, place it below the table using display math.

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

Final output rules:
- Never return broken Markdown tables.
- Never return table rows in one single line.
- Never return raw LaTeX without $...$ or $$...$$.
- Make the final answer attractive and readable for a downloaded study-notes PDF.
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

Summarize the following PDF content in simple student-friendly language.

Make the answer:
- Pointwise
- Easy to understand
- Useful for exam revision
- Include definitions, formulas, and important concepts
- Use proper unit-wise headings
- Explain formulas clearly
- Include important exam notes
- Add diagram-to-draw boxes wherever diagrams are needed
- Use proper Markdown tables wherever content is tabular
- Do not convert tables into bullet points
- Do not copy long paragraphs directly

Very important:
- Tables must be real Markdown tables.
- Every table row must be on a new line.
- Important equations must be in display math using $$...$$.

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

Rules:
- Use question boxes for important questions.
- Add short hints below difficult questions.
- Add formulas where needed.
- Add diagram-to-draw boxes for theory questions needing diagrams.
- Use proper Markdown tables for comparisons.

For numerical/formula-based questions:
- Write formulas using proper LaTeX.
- Put important formulas in display format using $$...$$.

{COMMON_RULES}

Content:
{text}
"""

    elif output_type == "MCQs":
        return f"""
Create 20 multiple choice questions from the following content.

Rules:
- Give 4 options for each question.
- Mark the correct answer.
- Add short explanation.
- Use simple exam-level language.
- If formula-based MCQs are present, write equations using LaTeX math format.
- Use clean numbering.
- Do not make the answer too lengthy.

{COMMON_RULES}

Content:
{text}
"""

    elif output_type == "Formula Sheet":
        return f"""
Create a clean formula sheet from the following content.

Rules:
- Extract all important formulas.
- Write every important formula in display LaTeX format using $$...$$.
- Explain symbols below each formula.
- Add where each formula is used.
- Keep it short and exam-oriented.
- Use proper Markdown tables for compact formula summaries.
- Do not convert tables into bullet points.
- If the PDF contains a table, convert it into a clean Markdown table.
- Write one major formula per line, not side by side.
- Make equations look like textbook-style equations.
- Add note boxes for common exam mistakes.

Use this format:

## Formula Name

$$
formula here
$$

Where:
- $symbol$ = meaning

Used for: explanation.

> **Note:** Important exam tip here.

{COMMON_RULES}

Content:
{text}
"""

    elif output_type == "Viva Questions":
        return f"""
Create viva questions and answers from the following content.

Rules:
- Make questions simple.
- Give short answers.
- Focus on exam and oral viva preparation.
- Use student-friendly language.
- If any answer contains a formula, write it using LaTeX math format.
- Use clean numbering.
- Add important viva tips as note boxes.

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
- Extract important concepts.
- Extract definitions.
- Extract formulas.
- Extract tables if they are important.
- Keep it concise.
- Use bullet points where needed.
- Use proper LaTeX for equations.
- Use proper Markdown tables when content is tabular.
- Do not convert tables into bullet points.
- Every Markdown table row must be on a new line.
- Add note boxes for exam tips.
- Add diagram-to-draw boxes if a diagram is needed.
- Do not make long explanations.

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
- Remove repetition.
- Organize properly with headings.
- Keep it exam-oriented.
- Use simple student-friendly language.
- Preserve important formulas.
- Use clean LaTeX equations.
- Use clean Markdown tables when useful, especially for:
  - Truth tables
  - Comparison tables
  - Formula summaries
  - Operators and observables
  - Advantages vs disadvantages
- Do not convert tables into bullet points.
- Every Markdown table row must be on a separate new line.
- If table-like data exists, convert it into a proper Markdown table.
- Add exam question boxes where needed.
- Add note boxes for important tips.
- Add diagram-to-draw boxes wherever diagrams are required.
- Make the final answer look like a professional exam-prep PDF.

Very strict table rule:
Never write this:
| A | B | |---|---| | X | Y |

Always write this:

| A | B |
|---|---|
| X | Y |

Very strict equation rule:
Never write important formulas inside a sentence.
Always write important equations like this:

$$
equation here
$$

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

Very important formatting rules:
- Always write mathematical equations using KaTeX-compatible LaTeX.
- Use $...$ only for short inline equations.
- Use $$...$$ for all important equations.
- Never use \\[ \\] or \\( \\).
- Never write equations as raw plain text.
- Never start equations with square brackets like [i\\hbar.
- Use Markdown tables when they improve readability.
- Every Markdown table row must be on a separate new line.
- Never write a full table in one single line.
- Keep tables short and properly formatted.
- Use note boxes and exam question boxes using Markdown blockquotes.
- Make output suitable for an attractive downloaded study-notes PDF.
"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.15,
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


def repair_one_line_tables(text):
    """
    Attempts to repair tables that AI accidentally returns in one line.
    Example:
    | A | B | |---|---| | X | Y |
    becomes:
    | A | B |
    |---|---|
    | X | Y |
    """

    repaired_lines = []

    for line in text.split("\n"):
        stripped = line.strip()

        # Only repair suspicious one-line tables
        if stripped.startswith("|") and stripped.count("|") >= 8:
            # Split table rows at places like | | or || with optional spaces
            rows = re.split(r"\|\s*\|", stripped)

            clean_rows = []
            for row in rows:
                row = row.strip()

                if not row:
                    continue

                if not row.startswith("|"):
                    row = "| " + row

                if not row.endswith("|"):
                    row = row + " |"

                # Normalize spacing around pipes
                row = re.sub(r"\s*\|\s*", " | ", row)
                row = row.replace("|  |", "|")
                row = row.strip()

                clean_rows.append(row)

            # If repair gives multiple rows, use them
            if len(clean_rows) >= 2:
                repaired_lines.append("")
                repaired_lines.extend(clean_rows)
                repaired_lines.append("")
            else:
                repaired_lines.append(line)
        else:
            repaired_lines.append(line)

    return "\n".join(repaired_lines)


def normalize_markdown_tables(text):
    """
    Ensures blank lines around Markdown tables for proper frontend rendering.
    """
    lines = text.split("\n")
    output = []
    in_table = False

    for i, line in enumerate(lines):
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


def normalize_equations(text):
    """
    Cleans common equation issues and promotes important inline LaTeX to display math.
    """
    text = text.replace("\\[", "$$")
    text = text.replace("\\]", "$$")
    text = text.replace("\\(", "$")
    text = text.replace("\\)", "$")

    # Promote important inline equations to display math if they contain major LaTeX commands
    important_latex = r"(\\frac|\\partial|\\int|\\sum|\\nabla|\\hat|\\sqrt|\\Psi|\\psi|\\hbar)"

    def replace_inline_math(match):
        equation = match.group(1).strip()

        if re.search(important_latex, equation) and len(equation) > 12:
            return f"\n\n$$\n{equation}\n$$\n\n"

        return f"${equation}$"

    text = re.sub(r"\$([^$\n]+)\$", replace_inline_math, text)

    return text


def clean_ai_output(text):
    """
    Final cleanup before sending AI output to frontend.
    """
    text = repair_one_line_tables(text)
    text = normalize_markdown_tables(text)
    text = normalize_equations(text)

    # Remove excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


def process_pdf_chunks(chunks, output_type):
    if not chunks:
        raise Exception("No readable text found in this PDF")

    # Small PDF: direct generation
    if len(chunks) == 1:
        prompt = create_prompt(chunks[0], output_type)
        output = generate_with_groq(prompt, max_tokens=2200)
        return clean_ai_output(output)

    # Large PDF: chunk processing
    partial_summaries = []

    for index, chunk in enumerate(chunks):
        chunk_prompt = create_chunk_prompt(chunk, index + 1, len(chunks))

        # Balanced output size for speed + quality
        partial_summary = generate_with_groq(chunk_prompt, max_tokens=700)
        partial_summary = clean_ai_output(partial_summary)

        partial_summaries.append(f"## Part {index + 1}\n\n{partial_summary}")

        # Smaller delay to reduce waiting time
        time.sleep(0.5)

    combined_notes = "\n\n".join(partial_summaries)
    combined_notes = combined_notes[:9000]

    final_prompt = create_final_prompt(combined_notes, output_type)

    # Balanced final output size
    final_output = generate_with_groq(final_prompt, max_tokens=2200)
    final_output = clean_ai_output(final_output)

    return final_output


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ExamEase AI backend is running with improved table, color-ready, and equation formatting support"
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