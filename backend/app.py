from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import os
import time
import re
import base64
from dotenv import load_dotenv
from groq import Groq

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "llama-3.1-8b-instant"

MAX_CHUNK_CHARS = 3500
MAX_CHUNKS = 6
MAX_SOURCE_IMAGES = 4


COMMON_RULES = """
You must create clean exam-ready content that will be rendered as Markdown and downloaded as a PDF.

GENERAL RULES:
- Output must be valid Markdown.
- Keep language simple and student-friendly.
- Use proper headings.
- Do not generate broken Markdown.
- Do not repeat the same points again and again.
- Do not add irrelevant formulas from other subjects.
- Only use formulas, concepts, examples, and diagrams that match the uploaded PDF content.

TABLE RULES:
- Use proper Markdown tables only when the content is truly tabular.
- Never write a table in one single line.
- Every table row must be on a separate line.
- Every table row must start with | and end with |.
- Always keep a blank line before and after a table.
- Do not use display equations inside table cells.
- Do not put long formulas inside table cells.
- Inside table cells, use only short inline math.
- If a formula is long, write "See formula below" in the table cell and put the full formula below the table.
- Do not leave incomplete rows like "| Momentum |".

EQUATION RULES:
- Use KaTeX-compatible LaTeX.
- Use $...$ only for small inline expressions.
- Use $$...$$ for important or long equations.
- Do not use \\[ \\].
- Do not use \\( \\).
- Do not write lone $ symbols.
- Do not write equations as plain broken text.
- Put every important equation on its own display block.

SOLVED EXAMPLE RULES:
Always write examples in this format:

### Example Title

**Given:**
- Given data or expression

**Solution:**
1. Step one
2. Step two
3. Step three

**Final Answer:**
Final answer here.

FINAL CHECK:
- No lone $ symbols.
- No one-line tables.
- No broken table rows.
- No raw LaTeX outside $...$ or $$...$$.
"""


def create_chunks_from_pdf(file_path):
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


def extract_source_images(file_path):
    """
    Extracts real embedded images from the uploaded PDF.
    If the PDF has no embedded images, it returns an empty list.
    """
    images = []

    if fitz is None:
        return images

    try:
        doc = fitz.open(file_path)

        for page_index in range(len(doc)):
            if len(images) >= MAX_SOURCE_IMAGES:
                break

            page = doc[page_index]
            page_images = page.get_images(full=True)

            for img_index, img in enumerate(page_images):
                if len(images) >= MAX_SOURCE_IMAGES:
                    break

                xref = img[0]

                try:
                    pix = fitz.Pixmap(doc, xref)

                    if pix.n >= 5:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    image_bytes = pix.tobytes("png")

                    # Skip very tiny icons/logos
                    if len(image_bytes) < 8000:
                        continue

                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                    images.append({
                        "page": page_index + 1,
                        "src": f"data:image/png;base64,{image_base64}"
                    })

                    pix = None

                except Exception:
                    continue

        doc.close()

    except Exception:
        return []

    return images


def create_prompt(text, output_type):
    if output_type == "Summary":
        return create_summary_prompt(text)

    if output_type == "Important Questions":
        return create_important_questions_prompt(text)

    if output_type == "MCQs":
        return create_mcq_prompt(text)

    if output_type == "Formula Sheet":
        return create_formula_sheet_prompt(text)

    if output_type == "Viva Questions":
        return create_viva_prompt(text)

    return create_summary_prompt(text)


def create_summary_prompt(text):
    return f"""
You are an exam preparation assistant.

Create a LONG DETAILED SUMMARY from the uploaded PDF content.

IMPORTANT:
- This must be proper study notes, not a question bank.
- The output must be long enough to become minimum 5 PDF pages.
- Target 1600 to 2200 words if enough content exists.
- Do not make short sections.
- Explain every topic clearly.
- Add examples wherever possible.
- Add tables wherever useful.
- Do not add irrelevant content outside the PDF topic.

STRICT SUMMARY FORMAT:

# Detailed Summary Notes

## UNIT / TOPIC NAME

### 1. Introduction
Write a detailed introduction in 8 to 12 lines.

### 2. Important Concepts
Explain each important concept in detail.
Use bullet points, but each point must have explanation.

### 3. Definitions
Give important definitions clearly.
Each definition should be exam-ready.

### 4. Detailed Explanation of Topics
Explain every major subtopic one by one.
Add subheadings.
Give detailed explanation, not just one-line points.

### 5. Important Formulas / Laws / Rules
Write all formulas, laws, rules, expressions, or theorems present in the PDF.
Use display math for important formulas.

### 6. Important Tables
Use tables for comparisons, truth tables, laws, gates, properties, or classifications.

### 7. Solved / Explanation Examples
Add solved examples if present.
Use this format only:
Given:
Solution:
Final Answer:

### 8. Exam Revision Box
Put all notes, diagram points, exam questions, and revision points inside ONE single blockquote box.

Use exactly this style:

> **Important Notes:**
> - Point 1
> - Point 2
> - Point 3
>
> **Diagrams to Practice:**
> - Diagram 1
> - Diagram 2
>
> **Important Exam Questions:**
> - Question 1
> - Question 2
>
> **Quick Revision Points:**
> - Revision point 1
> - Revision point 2
> - Revision point 3

DO NOT:
- Do not create many separate note boxes.
- Do not create many separate diagram boxes.
- Do not create many separate revision boxes.
- Do not create 2-mark, 5-mark, 10-mark sections.
- Do not create MCQs.
- Do not create a question bank.
- Do not write headings like "Formula Below".
- Do not write "Long Formulas".
- Do not keep the summary under 5 pages.

{COMMON_RULES}

PDF Content:
{text}
"""


def create_important_questions_prompt(text):
    return f"""
You are a college exam question paper expert.

Create ONLY an IMPORTANT QUESTIONS document from the PDF content.

This must look like a real question bank, not a summary.

STRICT IMPORTANT QUESTIONS FORMAT:

# Important Questions

## UNIT / TOPIC NAME

## 2-Mark Questions

Q1. Write the question.
Answer: Give a short 2-3 line answer.

Q2. Write the question.
Answer: Give a short 2-3 line answer.

Q3. Write the question.
Answer: Give a short 2-3 line answer.

Q4. Write the question.
Answer: Give a short 2-3 line answer.

## 5-Mark Questions

Q1. Write the question.
Answer points:
- Point 1
- Point 2
- Point 3
- Point 4
- Add formula/table if needed.

Q2. Write the question.
Answer points:
- Point 1
- Point 2
- Point 3
- Point 4

## 10-Mark Questions

Q1. Write the question.
Answer outline:
- Introduction
- Main explanation
- Important laws/formulas/table
- Diagram to draw if required
- Conclusion

Q2. Write the question.
Answer outline:
- Introduction
- Main explanation
- Important laws/formulas/table
- Diagram to draw if required
- Conclusion

## Numericals / Simplification / Truth Table Questions

Q1. Write the question.
Solution:
- Step 1
- Step 2
- Final answer

Q2. Write the question.
Solution:
- Step 1
- Step 2
- Final answer

## Viva Questions

Q1. Question?
Ans: Short oral answer.

Q2. Question?
Ans: Short oral answer.

Q3. Question?
Ans: Short oral answer.

VERY IMPORTANT:
- Do not write "Important Concepts" as a main section.
- Do not write "Definitions" as a main section.
- Do not write "Notes" as a main section.
- Do not write "Important Tables" as a main section.
- Do not write "Diagram to Draw" as a separate main section.
- Tables, formulas, and diagrams can appear only inside answers where needed.
- Every question must have an answer or answer outline.
- Remove repeated questions.
- This should look completely different from Summary.

{COMMON_RULES}

PDF Content:
{text}
"""


def create_mcq_prompt(text):
    return f"""
Create ONLY MCQs from the PDF content.

STRICT MCQ FORMAT:

# MCQ Practice Set

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
- Do not write theory notes.
- Do not create 2-mark, 5-mark, or 10-mark sections.
- Keep options simple and exam-level.

{COMMON_RULES}

PDF Content:
{text}
"""


def create_formula_sheet_prompt(text):
    return f"""
Create ONLY a COMPACT 16:9 FORMULA SHEET from the PDF content.

This should look like a professional cheat sheet, not normal notes.

IMPORTANT:
- Use dense card-based sections.
- Each card should contain compact formulas, laws, rules, truth tables, diagram hints, and exam use.
- Do not write long paragraphs.
- Do not create generic cards if specific cards are possible.
- Use only formulas/laws/rules from the uploaded PDF.
- Do not add unrelated formulas.
- Keep every card balanced and compact.
- Avoid very large cards.
- If one topic is too large, split it into two smaller cards.

For Digital Electronics, prefer cards like:
- Basic Logic Gates
- NOT / Inverter Gate
- AND / OR Gates
- Universal Gates
- Boolean Laws
- De Morgan's Theorems
- Truth Tables
- Simplification Rules
- SOP / POS Forms
- Important Exam Formula Table

STRICT FORMULA SHEET FORMAT:

# UNIT FORMULA SHEET TITLE

**Topic 1 • Topic 2 • Topic 3 • Important Laws • Tables • Diagrams**

## CARD 1: Specific Topic Name

**Diagram:** Short diagram instruction if required.

**Formula / Rule:**
- Short formula or rule 1
- Short formula or rule 2

**Important Points:**
- Point 1
- Point 2
- Point 3

**Exam Use:** One short line.

> **Note:** One short exam tip.

## CARD 2: Specific Topic Name

Continue same compact card format.

## QUICK FORMULAS / LAWS

| Formula / Law | Expression | Use |
|---|---|---|
| Law name | Short expression | Use |

RULES:
- Generate 8 to 10 compact cards if enough content is available.
- Each card must be compact.
- Use tables for truth tables, Boolean laws, comparisons, or properties.
- Do not create huge law cards.
- Do not create Summary sections.
- Do not create Important Questions sections.
- Do not create MCQs.
- Do not create Viva Questions.
- This output will be shown in landscape PDF.

{COMMON_RULES}

PDF Content:
{text}
"""


def create_viva_prompt(text):
    return f"""
Create ONLY VIVA QUESTIONS AND ANSWERS from the PDF content.

STRICT VIVA FORMAT:

# Viva Questions and Answers

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
- Focus on quick viva revision.

{COMMON_RULES}

PDF Content:
{text}
"""


def create_chunk_prompt(chunk_text, chunk_number, total_chunks, output_type):
    if output_type == "Summary":
        return f"""
You are reading part {chunk_number} of {total_chunks} from a college PDF.

Extract DETAILED SUMMARY MATERIAL only.

Return:
- Important concepts with explanation
- Definitions
- Detailed explanations
- Important formulas/laws/rules
- Useful tables
- Solved examples in Given/Solution/Final Answer format
- Diagram points
- Short exam notes
- Quick revision points

Do NOT create a question bank.
Make this detailed enough for long summary notes.

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""

    if output_type == "Important Questions":
        return f"""
You are reading part {chunk_number} of {total_chunks} from a college PDF.

Extract ONLY exam questions from this part.

Return:
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

## Possible Numericals / Simplification / Truth Table Questions
Q. Question?
Solution:
- Step 1
- Step 2

## Possible Viva Questions
Q. Question?
Ans: Short answer.

Do NOT write summary sections.

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""

    if output_type == "MCQs":
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

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""

    if output_type == "Formula Sheet":
        return f"""
You are reading part {chunk_number} of {total_chunks} from a college PDF.

Extract ONLY compact formula-sheet material.

Return compact cards:
## CARD: Specific Topic Name

**Formula / Rule:**
- Short formula or rule 1
- Short formula or rule 2

**Important Points:**
- Point 1
- Point 2
- Point 3

**Truth Table / Law Table:** Add only if useful.

**Exam Use:** One short line.

Do not write long notes.
Do not create question answers.
Prefer specific cards over generic cards.
Avoid huge cards.

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""

    if output_type == "Viva Questions":
        return f"""
You are reading part {chunk_number} of {total_chunks} from a college PDF.

Extract ONLY viva questions and short answers.

Format:
Q1. Question?
Ans: Short answer.

Q2. Question?
Ans: Short answer.

{COMMON_RULES}

PDF Part {chunk_number}/{total_chunks}:
{chunk_text}
"""

    return create_chunk_prompt(chunk_text, chunk_number, total_chunks, "Summary")


def create_final_prompt(combined_notes, output_type):
    if output_type == "Summary":
        return f"""
You are an exam preparation assistant.

Combine the partial notes into one final LONG DETAILED SUMMARY.

The final summary must be detailed enough to become minimum 5 PDF pages.

STRICT FINAL SUMMARY FORMAT:

# Detailed Summary Notes

## UNIT / TOPIC NAME

### 1. Introduction
### 2. Important Concepts
### 3. Definitions
### 4. Detailed Explanation of Topics
### 5. Important Formulas / Laws / Rules
### 6. Important Tables
### 7. Solved / Explanation Examples
### 8. Exam Revision Box

Rules:
- Target 1600 to 2200 words if enough content is available.
- Add enough explanation under each heading.
- Explain subtopics properly.
- Use clean solved example format: Given, Solution, Final Answer.
- For section 8, put all notes, diagrams, exam questions, and revision points inside ONE single blockquote box.
- Do not create many separate yellow/orange boxes.
- Do not create 2-mark, 5-mark, 10-mark sections.
- Do not create a question bank.
- Do not create MCQs.
- Do not write "Formula Below".
- Do not write "Long Formulas".
- Remove repetition.

Use this exact style for Section 8:

### 8. Exam Revision Box

> **Important Notes:**
> - Point 1
> - Point 2
> - Point 3
>
> **Diagrams to Practice:**
> - Diagram 1
> - Diagram 2
>
> **Important Exam Questions:**
> - Question 1
> - Question 2
>
> **Quick Revision Points:**
> - Revision point 1
> - Revision point 2
> - Revision point 3

{COMMON_RULES}

Partial Notes:
{combined_notes}
"""

    if output_type == "Important Questions":
        return f"""
You are a college exam question paper expert.

Combine the partial notes into one final IMPORTANT QUESTIONS document.

STRICT FINAL IMPORTANT QUESTIONS FORMAT:

# Important Questions

## UNIT / TOPIC NAME

## 2-Mark Questions
Q1. Question?
Answer: Short answer.

Q2. Question?
Answer: Short answer.

Q3. Question?
Answer: Short answer.

Q4. Question?
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

## Numericals / Simplification / Truth Table Questions
Q1. Question?
Solution:
- Step 1
- Step 2
- Final answer

Q2. Question?
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

{COMMON_RULES}

Partial Question Material:
{combined_notes}
"""

    if output_type == "MCQs":
        return f"""
Combine the partial MCQs into one final MCQ set.

STRICT FINAL MCQ FORMAT:

# MCQ Practice Set

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

    if output_type == "Formula Sheet":
        return f"""
Combine the partial formula material into one final COMPACT 16:9 FORMULA SHEET.

The output must look like a clean cheat sheet with multiple compact cards.

STRICT FINAL FORMULA SHEET FORMAT:

# UNIT FORMULA SHEET TITLE

**Topic 1 • Topic 2 • Topic 3 • Laws • Tables • Diagrams**

## CARD 1: Specific Topic Name

**Diagram:** Short diagram instruction if needed.

**Formula / Rule:**
- Short formula or rule 1
- Short formula or rule 2

**Important Points:**
- Point 1
- Point 2
- Point 3

**Truth Table / Law Table:** Add only if useful.

**Exam Use:** One short line.

> **Note:** One short exam tip.

## CARD 2: Specific Topic Name

Continue same compact format.

## QUICK FORMULAS / LAWS

| Formula / Law | Expression | Use |
|---|---|---|
| Law name | Short expression | Use |

Rules:
- Generate 8 to 10 compact cards if content allows.
- Keep every card compact.
- Prefer specific cards over generic cards.
- Split large topics into smaller cards.
- Do not create very large cards.
- Use tables for truth tables, Boolean laws, formula summaries, and comparisons.
- Do not create summary paragraphs.
- Do not create question bank.
- Do not create MCQs.
- Do not create viva questions.
- The content will be displayed in landscape PDF.

{COMMON_RULES}

Partial Formula Material:
{combined_notes}
"""

    if output_type == "Viva Questions":
        return f"""
Combine the partial viva questions into one final VIVA QUESTIONS document.

STRICT FINAL VIVA FORMAT:

# Viva Questions and Answers

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

{COMMON_RULES}

Partial Viva Material:
{combined_notes}
"""

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
- Summary must look like detailed summary notes.
- Important Questions must look like a real question bank.
- MCQs must contain only MCQs.
- Formula Sheet must look like a compact cheat-sheet with cards.
- Viva Questions must contain only viva Q&A.
- Output must be valid Markdown.
- Use Markdown tables correctly.
- Every Markdown table row must be on a new line.
- Never write a full table in one line.
- Never put display equations inside table cells.
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
                fixed_rows.append(row.strip())

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


def clean_ai_output(text, output_type):
    text = normalize_math_delimiters(text)
    text = repair_one_line_tables(text)
    text = protect_tables_from_long_equations(text)
    text = normalize_markdown_tables(text)
    text = remove_orphan_dollars(text)
    text = remove_broken_pipe_lines(text)

    if output_type == "Summary":
        text = re.sub(r"(?i)^#+\s*formula below\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"(?i)^#+\s*long formulas\s*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


def process_pdf_chunks(chunks, output_type):
    if not chunks:
        raise Exception("No readable text found in this PDF")

    if len(chunks) == 1:
        prompt = create_prompt(chunks[0], output_type)

        if output_type == "Summary":
            output = generate_with_groq(prompt, max_tokens=5000)
        elif output_type == "Formula Sheet":
            output = generate_with_groq(prompt, max_tokens=3600)
        else:
            output = generate_with_groq(prompt, max_tokens=2800)

        return clean_ai_output(output, output_type)

    partial_outputs = []

    for index, chunk in enumerate(chunks):
        chunk_prompt = create_chunk_prompt(
            chunk,
            index + 1,
            len(chunks),
            output_type
        )

        if output_type == "Summary":
            partial_output = generate_with_groq(chunk_prompt, max_tokens=1300)
        elif output_type == "Formula Sheet":
            partial_output = generate_with_groq(chunk_prompt, max_tokens=950)
        else:
            partial_output = generate_with_groq(chunk_prompt, max_tokens=850)

        partial_output = clean_ai_output(partial_output, output_type)
        partial_outputs.append(f"## Part {index + 1}\n\n{partial_output}")

        time.sleep(0.5)

    combined_notes = "\n\n".join(partial_outputs)
    combined_notes = combined_notes[:14000]

    final_prompt = create_final_prompt(combined_notes, output_type)

    if output_type == "Summary":
        final_output = generate_with_groq(final_prompt, max_tokens=5000)
    elif output_type == "Formula Sheet":
        final_output = generate_with_groq(final_prompt, max_tokens=3600)
    else:
        final_output = generate_with_groq(final_prompt, max_tokens=3000)

    final_output = clean_ai_output(final_output, output_type)

    return final_output


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ExamEase AI backend is running with source image extraction and improved layouts"
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

        source_images = []
        if output_type == "Summary":
            source_images = extract_source_images(file_path)

        return jsonify({
            "message": "AI notes generated successfully",
            "text": ai_output,
            "images": source_images
        })

    except Exception as e:
        error_message = str(e)

        if "rate_limit_exceeded" in error_message or "tokens per minute" in error_message:
            return jsonify({
                "error": "The PDF is large or the free AI token limit was reached. Please wait 1 minute and try again."
            }), 500

        if "Request too large" in error_message:
            return jsonify({
                "error": "This PDF is too large for the current free AI limit. Try uploading unit-wise PDFs."
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