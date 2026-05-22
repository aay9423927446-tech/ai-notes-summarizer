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

MAX_CHUNK_CHARS = 3200
MAX_CHUNKS = 4
MAX_SOURCE_IMAGES = 3


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
- Do not put long display formulas inside table cells.
- Inside table cells, use only short inline math using $...$.
- Do not write raw LaTeX like \\frac, \\left, \\right outside $...$.
- If a formula is too long, write it below the table using $$...$$.
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

            for img in page_images:
                if len(images) >= MAX_SOURCE_IMAGES:
                    break

                xref = img[0]

                try:
                    pix = fitz.Pixmap(doc, xref)

                    if pix.n >= 5:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    image_bytes = pix.tobytes("png")

                    # Skip tiny icons/logos
                    if len(image_bytes) < 10000:
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

Create a DETAILED SUMMARY from the uploaded PDF content.

IMPORTANT:
- This must be proper study notes, not a question bank.
- Do not include important exam questions in Summary.
- We already have a separate Important Questions output type.
- The output should be detailed and exam-oriented.
- Target around 1200 to 1600 words if enough content exists.
- Do not make short sections.
- Explain every topic clearly.
- Add examples wherever possible.
- Add tables wherever useful.
- Do not add irrelevant content outside the PDF topic.
- Do not create a separate source image section.
- Do not write raw HTML.

STRICT SUMMARY FORMAT:

# Detailed Summary Notes

## UNIT / TOPIC NAME

### 1. Introduction
Write a detailed introduction.

### 2. Important Concepts
Explain each important concept clearly.

### 3. Definitions
Give important definitions clearly.

### 4. Detailed Explanation of Topics
Explain major subtopics one by one.

### 5. Important Formulas / Laws / Rules
Write all formulas, laws, rules, expressions, or theorems present in the PDF.

### 6. Important Tables
Use tables for comparisons, truth tables, laws, gates, properties, or classifications.

### 7. Solved / Explanation Examples
Add solved examples if present.
Use this format:
Given:
Solution:
Final Answer:

### 8. Exam Revision Box
Put only notes, diagram points, and revision points inside ONE single blockquote box.

Use exactly this style:

> **Important Notes:**
> - Point 1
> - Point 2
>
> **Diagrams to Practice:**
> - Diagram 1
> - Diagram 2
>
> **Quick Revision Points:**
> - Revision point 1
> - Revision point 2

DO NOT:
- Do not include Important Exam Questions in Summary.
- Do not create many separate note boxes.
- Do not create many separate diagram boxes.
- Do not create many separate revision boxes.
- Do not create a separate source image section.
- Do not create 2-mark, 5-mark, 10-mark sections.
- Do not create MCQs.
- Do not create a question bank.
- Do not write headings like "Formula Below".
- Do not write "Long Formulas".
- Do not write raw HTML tags.

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

## Viva Questions

Q1. Question?
Ans: Short oral answer.

Q2. Question?
Ans: Short oral answer.

VERY IMPORTANT:
- Do not write summary sections.
- Every question must have an answer or answer outline.
- Remove repeated questions.

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
- Create 20 MCQs if enough content exists.
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
Create ONLY a COMPACT FORMULA SHEET from the PDF content.

This should look like a professional cheat sheet, not normal notes.

IMPORTANT:
- Use dense card-based sections.
- Do not create empty cards.
- Every card must have a proper card title.
- Every card must contain at least 2 useful formulas, laws, rules, or key points.
- If the PDF does not contain enough formula content for a card, do not create that card.
- Do not leave empty labels like "Diagram:", "Formula / Rule:", "Important Points:" without content.
- Do not write long paragraphs.
- Use only formulas/laws/rules from the uploaded PDF.
- Do not add unrelated formulas.
- Keep every card balanced and compact.
- Avoid very large cards.
- Keep cards short so there are no long empty gaps.

FORMULA FORMAT RULES:
- Every mathematical formula must be inside $...$ or $$...$$.
- In tables, formulas must be short and written as inline math like $I = V/R$.
- Do not write raw LaTeX outside math delimiters.
- Do not use \\left and \\right unless inside $...$.
- Do not write things like V_D = V_T \\ln\\left(...) without $...$.

For Electronics, prefer cards like:
- Diode Equation
- P-N Junction Biasing
- Rectifier Formulas
- Zener Diode
- Op-Amp Basics
- Inverting Op-Amp
- Differential Amplifier
- Important Formula Table

STRICT FORMULA SHEET FORMAT:

# UNIT FORMULA SHEET TITLE

**Topic 1 • Topic 2 • Topic 3 • Important Laws • Tables • Diagrams**

## CARD 1: Specific Topic Name

**Formula / Rule:**
- $formula_1$
- $formula_2$

**Important Points:**
- Point 1
- Point 2

**Exam Use:** One short line.

> **Note:** One short exam tip.

## CARD 2: Specific Topic Name

Continue same compact card format.

## QUICK FORMULAS / LAWS

| Formula / Law | Expression | Use |
|---|---|---|
| Law name | $short formula$ | Use |

RULES:
- Generate 5 to 8 compact cards if enough content is available.
- Each card must contain real content.
- Do not create huge law cards.
- Do not create empty cards.
- Do not create Summary sections.
- Do not create Important Questions sections.
- Do not create MCQs.
- Do not create Viva Questions.

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
Do NOT include important exam questions.
Do NOT create separate image section.
Do NOT write raw HTML.
Keep it detailed but concise.

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
- $formula or rule 1$
- $formula or rule 2$

**Important Points:**
- Point 1
- Point 2

**Exam Use:** One short line.

Rules:
- Do not create empty cards.
- Do not create cards with only labels.
- Do not write long notes.
- Do not create question answers.
- Prefer specific cards over generic cards.
- Avoid huge cards.
- Every formula must be inside $...$ or $$...$$.

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

Combine the partial notes into one final DETAILED SUMMARY.

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
- Target around 1200 to 1600 words if enough content is available.
- Add enough explanation under each heading.
- Explain subtopics properly.
- Use clean solved example format: Given, Solution, Final Answer.
- For section 8, put only notes, diagrams, and revision points inside ONE single blockquote box.
- Do not include Important Exam Questions in Summary.
- Do not create many separate yellow/orange boxes.
- Do not create separate source image section.
- Do not create 2-mark, 5-mark, 10-mark sections.
- Do not create a question bank.
- Do not create MCQs.
- Do not write "Formula Below".
- Do not write "Long Formulas".
- Do not write raw HTML.
- Remove repetition.

Use this exact style for Section 8:

### 8. Exam Revision Box

> **Important Notes:**
> - Point 1
> - Point 2
>
> **Diagrams to Practice:**
> - Diagram 1
> - Diagram 2
>
> **Quick Revision Points:**
> - Revision point 1
> - Revision point 2

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

## Numericals / Simplification / Truth Table Questions
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
- Do not write summary sections.
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
Combine the partial formula material into one final COMPACT FORMULA SHEET.

STRICT FINAL FORMULA SHEET FORMAT:

# UNIT FORMULA SHEET TITLE

**Topic 1 • Topic 2 • Topic 3 • Laws • Tables • Diagrams**

## CARD 1: Specific Topic Name

**Formula / Rule:**
- $formula or rule 1$
- $formula or rule 2$

**Important Points:**
- Point 1
- Point 2

**Exam Use:** One short line.

> **Note:** One short exam tip.

## CARD 2: Specific Topic Name

Continue same compact format.

## QUICK FORMULAS / LAWS

| Formula / Law | Expression | Use |
|---|---|---|
| Law name | $short formula$ | Use |

Rules:
- Generate 5 to 8 compact cards if content allows.
- Keep every card compact.
- Prefer specific cards over generic cards.
- Split large topics into smaller cards.
- Do not create very large cards.
- Do not create empty cards.
- Do not leave labels empty.
- Every formula must be inside $...$ or $$...$$.
- In tables, expressions must be short inline formulas like $I = V/R$.
- Do not write raw LaTeX outside math delimiters.
- Use tables for formula summaries and comparisons.
- Do not create summary paragraphs.
- Do not create question bank.
- Do not create MCQs.
- Do not create viva questions.

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
- Remove repeated questions.

{COMMON_RULES}

Partial Viva Material:
{combined_notes}
"""

    return create_final_prompt(combined_notes, "Summary")


def generate_with_groq(prompt, max_tokens=1200):
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
- Summary must NOT include important exam questions.
- Important Questions must look like a real question bank.
- MCQs must contain only MCQs.
- Formula Sheet must look like a compact cheat-sheet with cards.
- Formula Sheet must not contain empty cards.
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
- Never write raw HTML tags.
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


def wrap_latex_like_text(text):
    """
    Wraps common raw LaTeX fragments in inline math if the model forgot $...$.
    This mainly improves formula-sheet tables.
    """
    patterns = [
        r"(?<!\$)([A-Za-z]_[A-Za-z0-9]+\s*=\s*[^|\n]+\\frac[^|\n]+)(?!\$)",
        r"(?<!\$)([A-Za-z]\s*=\s*[^|\n]+\\frac[^|\n]+)(?!\$)",
        r"(?<!\$)([A-Za-z]_[A-Za-z0-9]+\s*=\s*[^|\n]+\\ln[^|\n]+)(?!\$)",
        r"(?<!\$)([A-Za-z]\s*=\s*[^|\n]+\\ln[^|\n]+)(?!\$)",
    ]

    for pattern in patterns:
        text = re.sub(pattern, r"$\1$", text)

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
                for token in ["\\frac", "\\partial", "\\int", "\\sum", "\\nabla", "\\left", "\\right"]
            )

            if has_long_formula and len(line) > 150:
                formulas = re.findall(r"\$([^$]+)\$", line)

                for formula in formulas:
                    if any(
                        token in formula
                        for token in ["\\frac", "\\partial", "\\int", "\\sum", "\\nabla", "\\left", "\\right"]
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


def remove_summary_questions(text):
    if "### 8. Exam Revision Box" not in text:
        return text

    text = re.sub(
        r">\s*\*\*Important Exam Questions:\*\*[\s\S]*?(?=>\s*\*\*Quick Revision Points:\*\*)",
        "",
        text
    )

    text = re.sub(
        r"\*\*Important Exam Questions:\*\*[\s\S]*?(?=\*\*Quick Revision Points:\*\*)",
        "",
        text
    )

    return text


def remove_empty_formula_cards(text):
    parts = re.split(r"(?=^## CARD\s*\d*[:.-])", text, flags=re.MULTILINE)

    if len(parts) <= 1:
        return text

    intro = parts[0]
    kept_cards = []

    for card in parts[1:]:
        card_lower = card.lower()

        useful_text = card
        useful_text = re.sub(r"## CARD\s*\d*[:.-].*", "", useful_text, flags=re.IGNORECASE)
        useful_text = useful_text.replace("**Diagram:**", "")
        useful_text = useful_text.replace("**Formula / Rule:**", "")
        useful_text = useful_text.replace("**Important Points:**", "")
        useful_text = useful_text.replace("**Exam Use:**", "")
        useful_text = useful_text.replace("**Note:**", "")
        useful_text = re.sub(r"[-•*\s:]+", " ", useful_text).strip()

        has_formula = "$" in card or "=" in card or "law" in card_lower or "rule" in card_lower
        has_enough_text = len(useful_text) > 80

        if has_formula and has_enough_text:
            kept_cards.append(card)

    return intro + "".join(kept_cards)


def clean_ai_output(text, output_type):
    text = normalize_math_delimiters(text)
    text = wrap_latex_like_text(text)
    text = repair_one_line_tables(text)
    text = protect_tables_from_long_equations(text)
    text = normalize_markdown_tables(text)
    text = remove_orphan_dollars(text)
    text = remove_broken_pipe_lines(text)

    # Remove any raw HTML accidentally generated or inserted
    text = re.sub(r"<\/?div[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\/?span[^>]*>", "", text, flags=re.IGNORECASE)

    if output_type == "Summary":
        text = re.sub(r"(?i)^#+\s*formula below\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"(?i)^#+\s*long formulas\s*$", "", text, flags=re.MULTILINE)
        text = remove_summary_questions(text)

    if output_type == "Formula Sheet":
        text = remove_empty_formula_cards(text)

    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


def process_pdf_chunks(chunks, output_type):
    if not chunks:
        raise Exception("No readable text found in this PDF")

    if len(chunks) == 1:
        prompt = create_prompt(chunks[0], output_type)

        if output_type == "Summary":
            output = generate_with_groq(prompt, max_tokens=3500)
        elif output_type == "Formula Sheet":
            output = generate_with_groq(prompt, max_tokens=2600)
        else:
            output = generate_with_groq(prompt, max_tokens=2200)

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
            partial_output = generate_with_groq(chunk_prompt, max_tokens=900)
        elif output_type == "Formula Sheet":
            partial_output = generate_with_groq(chunk_prompt, max_tokens=750)
        else:
            partial_output = generate_with_groq(chunk_prompt, max_tokens=700)

        partial_output = clean_ai_output(partial_output, output_type)
        partial_outputs.append(f"## Part {index + 1}\n\n{partial_output}")

        time.sleep(0.5)

    combined_notes = "\n\n".join(partial_outputs)
    combined_notes = combined_notes[:10000]

    final_prompt = create_final_prompt(combined_notes, output_type)

    if output_type == "Summary":
        final_output = generate_with_groq(final_prompt, max_tokens=3500)
    elif output_type == "Formula Sheet":
        final_output = generate_with_groq(final_prompt, max_tokens=2600)
    else:
        final_output = generate_with_groq(final_prompt, max_tokens=2200)

    final_output = clean_ai_output(final_output, output_type)

    return final_output


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ExamEase AI backend is running with fixed summary and formula formatting"
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
                "error": "The AI free token limit was reached. Please wait 30-60 seconds and try again, or upload a smaller/unit-wise PDF."
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