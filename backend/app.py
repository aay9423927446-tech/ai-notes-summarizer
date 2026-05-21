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
You must create clean exam-ready content that will be rendered as Markdown and downloaded as a PDF.

GENERAL FORMAT RULES:
- Output must be valid Markdown.
- Use clean headings.
- Keep language simple and exam-oriented.
- Do not write unnecessary long paragraphs.
- Do not generate broken Markdown.
- Do not repeat the same content again and again.

BOX FORMAT:
Use blockquotes only when needed.

For notes:
> **Note:** Write the important exam tip here.

For exam questions:
> **Exam Question:** Write the question here.

For diagrams:
> **Diagram to draw:** Describe the diagram clearly with labels.

TABLE RULES:
- Use proper Markdown tables only when the content is truly tabular.
- Do not write a table in one single line.
- Every table row must be on a separate line.
- Every table row must start with | and end with |.
- Always keep a blank line before and after a table.
- Do not use display equations inside table cells.
- Do not put long formulas inside table cells.
- Inside table cells, use only short inline math.
- If a formula is long, write "See formula below" in the table cell and put the full formula below the table.
- Do not leave incomplete table rows like "| Momentum |".

Correct table format:

| Quantity | Operator | Meaning |
|---|---|---|
| Position | $\\hat{x} = x$ | Measures position |
| Momentum | See formula below | Measures momentum |

Then write long formulas separately:

$$
\\hat{p} = -i\\hbar \\frac{\\partial}{\\partial x}
$$

EQUATION RULES:
- Use KaTeX-compatible LaTeX.
- Use $...$ only for small inline expressions.
- Use $$...$$ for important or long equations.
- Do not use \\[ \\].
- Do not use \\( \\).
- Do not write lone $ symbols.
- Do not write equations as plain broken text.
- Put every important equation on its own display block.

FINAL CHECK:
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


# -----------------------------
# DIRECT PROMPTS FOR SMALL PDFs
# -----------------------------

def create_prompt(text, output_type):
    if output_type == "Summary":
        return create_summary_prompt(text)

    elif output_type == "Important Questions":
        return create_important_questions_prompt(text)

    elif output_type == "MCQs":
        return create_mcq_prompt(text)

    elif output_type == "Formula Sheet":
        return create_formula_sheet_prompt(text)

    elif output_type == "Viva Questions":
        return create_viva_prompt(text)

    else:
        return create_summary_prompt(text)


def create_summary_prompt(text):
    return f"""
You are an exam preparation assistant.

Create a proper SUMMARY from the PDF content.

The output must be a study summary, not a question bank.

STRICT SUMMARY FORMAT:

## UNIT / TOPIC NAME

### 1. Introduction
- Give a short introduction.

### 2. Important Concepts
- Explain key concepts in bullet points.

### 3. Definitions
- Give important definitions.

### 4. Important Formulas
- Write formulas clearly.
- Use display math for important formulas.

### 5. Important Tables
- Add tables only if the PDF has useful tabular/comparison content.

### 6. Short Exam Notes
- Add short exam-oriented notes.

### 7. Diagram to Draw
> **Diagram to draw:** Mention only diagrams that are useful for exams.

DO NOT:
- Do not create 2-mark, 5-mark, 10-mark question sections.
- Do not make this look like a question paper.
- Do not include too many exam questions.
- At most include 2 important exam questions at the end.

{COMMON_RULES}

PDF Content:
{text}
"""


def create_important_questions_prompt(text):
    return f"""
You are a college exam question paper expert.

Create ONLY an IMPORTANT QUESTIONS document from the PDF content.

The output must look like a question bank, not a summary.

STRICT IMPORTANT QUESTIONS FORMAT:

## UNIT / TOPIC NAME

## 2-Mark Questions
Q1. Write the question.
Answer: Give a 2-3 line answer.

Q2. Write the question.
Answer: Give a 2-3 line answer.

## 5-Mark Questions
Q1. Write the question.
Answer points:
- Point 1
- Point 2
- Point 3
- Add formula/table if needed.

Q2. Write the question.
Answer points:
- Point 1
- Point 2
- Point 3

## 10-Mark Questions
Q1. Write the question.
Answer outline:
- Introduction
- Explanation
- Important formula/table
- Diagram to draw if required
- Conclusion

Q2. Write the question.
Answer outline:
- Introduction
- Explanation
- Important formula/table
- Diagram to draw if required
- Conclusion

## Numericals / Simplification Questions
Q1. Write numerical or simplification question if present.
Solution:
- Step 1
- Step 2
- Final answer

## Viva Questions
Q1. Question?
Ans: Short oral answer.

Q2. Question?
Ans: Short oral answer.

IMPORTANT:
- Do not write "Important Concepts" as a main section.
- Do not write "Definitions" as a main section.
- Do not write "Notes" as a main section.
- Do not write "Diagram to Draw" as a main section unless it is inside a question answer.
- Every question must have a short answer or answer outline.
- This should look different from Summary.

{COMMON_RULES}

PDF Content:
{text}
"""


def create_mcq_prompt(text):
    return f"""
Create ONLY MCQs from the PDF content.

STRICT MCQ FORMAT:

## UNIT / TOPIC NAME

1. Question text?
A. Option A
B. Option B
C. Option C
D. Option D

Correct Answer: C
Explanation: Short explanation.

2. Question text?
A. Option A
B. Option B
C. Option C
D. Option D

Correct Answer: A
Explanation: Short explanation.

Rules:
- Create 20 MCQs.
- Do not write summary.
- Do not write long theory answers.
- Do not add 2-mark, 5-mark, or 10-mark sections.
- Keep options simple and exam-level.
- If a formula is needed, use proper LaTeX.

{COMMON_RULES}

PDF Content:
{text}
"""


def create_formula_sheet_prompt(text):
    return f"""
Create ONLY a FORMULA SHEET from the PDF content.

STRICT FORMULA SHEET FORMAT:

## UNIT / TOPIC NAME

## Formula 1: Formula Name

$$
formula here
$$

Where:
- $symbol$ = meaning
- $symbol$ = meaning

Used for:
- Explain where this formula is used.

> **Note:** Important exam tip or common mistake.

## Formula 2: Formula Name

$$
formula here
$$

Where:
- $symbol$ = meaning

Used for:
- Usage explanation.

Rules:
- Do not write summary paragraphs.
- Do not create important questions.
- Do not create MCQs.
- Do not create viva questions.
- Focus only on formulas, symbols, and usage.
- Use tables only for short formula comparison if needed.

{COMMON_RULES}

PDF Content:
{text}
"""


def create_viva_prompt(text):
    return f"""
Create ONLY VIVA QUESTIONS AND ANSWERS from the PDF content.

STRICT VIVA FORMAT:

## UNIT / TOPIC NAME

Q1. What is ______?
Ans: Short and simple answer.

Q2. Define ______.
Ans: Short and simple answer.

Q3. Why is ______ important?
Ans: Short and simple answer.

Rules:
- Create short oral-answer style questions.
- Answers should be 1-3 lines.
- Do not write long theory.
- Do not write summary sections.
- Do not create 2-mark, 5-mark, 10-mark sections.
- Do not make tables unless absolutely needed.
- Focus on quick revision before viva.

{COMMON_RULES}

PDF Content:
{text}
"""


# -----------------------------
# CHUNK PROMPTS FOR LARGE PDFs
# -----------------------------

def create_chunk_prompt(chunk_text, chunk_number, total_chunks, output_type):
    if output_type == "Summary":
        return f"""
You are reading part {chunk_number} of {total_chunks} from a college PDF.

Extract only SUMMARY MATERIAL from this part.

Return:
- Important concepts
- Definitions
- Important formulas
- Useful tables
- Short exam notes
- Diagram points if required

Do NOT create a question bank.

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""

    elif output_type == "Important Questions":
        return f"""
You are reading part {chunk_number} of {total_chunks} from a college PDF.

Extract ONLY possible exam questions from this part.

Return questions in this rough format:

## Possible 2-Mark Questions
Q. Question?
Ans: Short answer.

## Possible 5-Mark Questions
Q. Question?
Answer points:
- Point 1
- Point 2
- Point 3

## Possible 10-Mark Questions
Q. Question?
Answer outline:
- Introduction
- Explanation
- Formula/table/diagram if needed
- Conclusion

## Possible Numericals / Simplification Questions
Q. Question?
Solution:
- Step 1
- Step 2

## Possible Viva Questions
Q. Question?
Ans: Short answer.

Do NOT write summary sections like Important Concepts, Definitions, Notes, or Diagram to Draw as main headings.

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""

    elif output_type == "MCQs":
        return f"""
You are reading part {chunk_number} of {total_chunks} from a college PDF.

Extract ONLY MCQs from this part.

Format:
1. Question?
A. Option A
B. Option B
C. Option C
D. Option D

Correct Answer: A
Explanation: Short explanation.

Do not write summary or theory notes.

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""

    elif output_type == "Formula Sheet":
        return f"""
You are reading part {chunk_number} of {total_chunks} from a college PDF.

Extract ONLY formulas from this part.

Format:
## Formula Name

$$
formula here
$$

Where:
- $symbol$ = meaning

Used for:
- Usage

Do not write summary or question answers.

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""

    elif output_type == "Viva Questions":
        return f"""
You are reading part {chunk_number} of {total_chunks} from a college PDF.

Extract ONLY viva questions and short answers from this part.

Format:
Q1. Question?
Ans: Short answer.

Q2. Question?
Ans: Short answer.

Do not write summary sections.

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""

    else:
        return create_chunk_prompt(chunk_text, chunk_number, total_chunks, "Summary")


# -----------------------------
# FINAL PROMPTS FOR LARGE PDFs
# -----------------------------

def create_final_prompt(combined_notes, output_type):
    if output_type == "Summary":
        return f"""
You are an exam preparation assistant.

Combine the partial notes into one final SUMMARY.

Final output must look like summary notes, not a question bank.

STRICT FINAL SUMMARY FORMAT:

## UNIT / TOPIC NAME

### 1. Introduction
### 2. Important Concepts
### 3. Definitions
### 4. Important Formulas
### 5. Important Tables
### 6. Short Exam Notes
### 7. Diagram to Draw

Rules:
- Do not create 2-mark, 5-mark, 10-mark sections.
- Do not create a question bank.
- At most include 2 important exam questions at the end.
- Remove repetition.
- Keep it clean and exam-oriented.

{COMMON_RULES}

Partial Notes:
{combined_notes}
"""

    elif output_type == "Important Questions":
        return f"""
You are a college exam question paper expert.

Combine the partial notes into one final IMPORTANT QUESTIONS document.

This must look completely different from a summary.

STRICT FINAL IMPORTANT QUESTIONS FORMAT:

## UNIT / TOPIC NAME

## 2-Mark Questions
Q1. Question?
Answer: Short answer.

Q2. Question?
Answer: Short answer.

## 5-Mark Questions
Q1. Question?
Answer points:
- Point 1
- Point 2
- Point 3
- Formula/table if needed

Q2. Question?
Answer points:
- Point 1
- Point 2
- Point 3

## 10-Mark Questions
Q1. Question?
Answer outline:
- Introduction
- Explanation
- Important laws/formulas
- Table if needed
- Diagram to draw if required
- Conclusion

Q2. Question?
Answer outline:
- Introduction
- Explanation
- Important laws/formulas
- Table if needed
- Diagram to draw if required
- Conclusion

## Numericals / Simplification Questions
Q1. Question?
Solution:
- Step 1
- Step 2
- Final answer

## Viva Questions
Q1. Question?
Ans: Short oral answer.

Q2. Question?
Ans: Short oral answer.

VERY IMPORTANT:
- Do not write "Important Concepts" section.
- Do not write "Definitions" section.
- Do not write "Notes" section.
- Do not write "Important Tables" section.
- Do not write "Diagram to Draw" as a separate main section.
- Tables/formulas/diagrams can appear only inside answers where needed.
- Every question must have an answer or answer outline.
- Remove repeated questions.
- Make it a real exam question bank.

{COMMON_RULES}

Partial Question Material:
{combined_notes}
"""

    elif output_type == "MCQs":
        return f"""
Combine the partial MCQs into one final MCQ set.

STRICT FINAL MCQ FORMAT:

## UNIT / TOPIC NAME

1. Question?
A. Option A
B. Option B
C. Option C
D. Option D

Correct Answer: A
Explanation: Short explanation.

Rules:
- Create exactly 20 MCQs if possible.
- Do not add summary sections.
- Do not add theory notes.
- Remove repeated MCQs.

{COMMON_RULES}

Partial MCQs:
{combined_notes}
"""

    elif output_type == "Formula Sheet":
        return f"""
Combine the partial formula notes into one final FORMULA SHEET.

STRICT FINAL FORMULA SHEET FORMAT:

## UNIT / TOPIC NAME

## Formula 1: Formula Name

$$
formula here
$$

Where:
- $symbol$ = meaning

Used for:
- Usage explanation.

> **Note:** Exam tip.

Rules:
- Do not add important questions.
- Do not add MCQs.
- Do not add viva questions.
- Do not add summary paragraphs.
- Only formulas, symbols, uses, and exam tips.

{COMMON_RULES}

Partial Formula Notes:
{combined_notes}
"""

    elif output_type == "Viva Questions":
        return f"""
Combine the partial viva questions into one final VIVA QUESTIONS document.

STRICT FINAL VIVA FORMAT:

## UNIT / TOPIC NAME

Q1. Question?
Ans: Short answer.

Q2. Question?
Ans: Short answer.

Q3. Question?
Ans: Short answer.

Rules:
- Answers must be short.
- Do not write summary sections.
- Do not add 2-mark, 5-mark, 10-mark sections.
- Remove repeated questions.
- Keep it useful for oral viva.

{COMMON_RULES}

Partial Viva Material:
{combined_notes}
"""

    else:
        return create_final_prompt(combined_notes, "Summary")


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
- Output must follow the selected output type exactly.
- Summary must look like summary notes.
- Important Questions must look like a question bank.
- MCQs must contain only MCQs.
- Formula Sheet must contain only formulas.
- Viva Questions must contain only viva Q&A.
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


# -----------------------------
# CLEANUP FUNCTIONS
# -----------------------------

def remove_orphan_dollars(text):
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        if line.strip() == "$":
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_math_delimiters(text):
    text = text.replace("\\[", "$$")
    text = text.replace("\\]", "$$")
    text = text.replace("\\(", "$")
    text = text.replace("\\)", "$")
    return text


def repair_one_line_tables(text):
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
    lines = text.split("\n")
    output = []
    delayed_formulas = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_line = (
            stripped.startswith("|")
            and stripped.endswith("|")
            and stripped.count("|") >= 2
        )

        if is_table_line:
            in_table = True

            has_long_formula = any(
                token in line
                for token in ["\\frac", "\\partial", "\\int", "\\sum", "\\nabla"]
            )

            if has_long_formula and len(line) > 120:
                formulas = re.findall(r"\$([^$]+)\$", line)

                for formula in formulas:
                    if any(
                        token in formula
                        for token in ["\\frac", "\\partial", "\\int", "\\sum", "\\nabla"]
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
    lines = text.split("\n")
    output = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_line = (
            stripped.startswith("|")
            and stripped.endswith("|")
            and stripped.count("|") >= 2
        )

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

    partial_outputs = []

    for index, chunk in enumerate(chunks):
        chunk_prompt = create_chunk_prompt(
            chunk,
            index + 1,
            len(chunks),
            output_type
        )

        partial_output = generate_with_groq(chunk_prompt, max_tokens=750)
        partial_output = clean_ai_output(partial_output)

        partial_outputs.append(f"## Part {index + 1}\n\n{partial_output}")

        time.sleep(0.5)

    combined_notes = "\n\n".join(partial_outputs)
    combined_notes = combined_notes[:9000]

    final_prompt = create_final_prompt(combined_notes, output_type)

    final_output = generate_with_groq(final_prompt, max_tokens=2400)
    final_output = clean_ai_output(final_output)

    return final_output


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ExamEase AI backend is running with distinct output types"
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