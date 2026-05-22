import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import "katex/dist/katex.min.css";
import "./App.css";

function App() {
  const [pdfFile, setPdfFile] = useState(null);
  const [outputType, setOutputType] = useState("Summary");
  const [result, setResult] = useState(
    "Your AI-generated notes will appear here."
  );
  const [sourceImages, setSourceImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("");

  const resultRef = useRef(null);

  const markdownPlugins = [remarkMath, remarkGfm];
  const rehypePlugins = [rehypeKatex];

  const handleFileChange = (e) => {
    setPdfFile(e.target.files[0]);
  };

  const fixOneLineTables = (text) => {
    let cleaned = text;

    cleaned = cleaned.replace(
      /(\|[^|\n]+(?:\|[^|\n]+)+\|)\s+(\|[-:\s|]+\|)\s+((?:\|[^|\n]+(?:\|[^|\n]+)+\|\s*)+)/g,
      (match, header, separator, rows) => {
        const fixedRows = rows
          .trim()
          .replace(/\|\s+\|/g, "|\n|")
          .replace(/\|\|/g, "|\n|");

        return `\n${header.trim()}\n${separator.trim()}\n${fixedRows}\n`;
      }
    );

    cleaned = cleaned.replace(/\|\s+\|/g, "|\n|");
    cleaned = cleaned.replace(/\|\|/g, "|\n|");

    return cleaned;
  };

  const removeOrphanDollars = (text) => {
    return text
      .split("\n")
      .filter((line) => line.trim() !== "$")
      .join("\n");
  };

  const normalizeMathDelimiters = (text) => {
    return text
      .replace(/\\\[/g, "$$")
      .replace(/\\\]/g, "$$")
      .replace(/\\\(/g, "$")
      .replace(/\\\)/g, "$");
  };

  const cleanBrokenMarkdown = (text) => {
    let cleaned = text;

    cleaned = cleaned.replace(/\n{4,}/g, "\n\n\n");
    cleaned = cleaned.replace(/\|\s*\n\s*\|/g, "|\n|");
    cleaned = cleaned.replace(/\n\s*\|\s*\n/g, "\n");
    cleaned = cleaned.replace(/\n\s*\|\s*$/gm, "");
    cleaned = cleaned.replace(/^\s*\|\s*$/gm, "");
    cleaned = cleaned.replace(/<\/?div[^>]*>/gi, "");
    cleaned = cleaned.replace(/<\/?span[^>]*>/gi, "");

    return cleaned;
  };

  const cleanMathOutput = (text) => {
    let cleaned = text;

    cleaned = normalizeMathDelimiters(cleaned);
    cleaned = fixOneLineTables(cleaned);
    cleaned = removeOrphanDollars(cleaned);
    cleaned = cleanBrokenMarkdown(cleaned);

    return cleaned.trim();
  };

  const handleGenerate = async () => {
    if (!pdfFile) {
      alert("Please upload a PDF file first");
      return;
    }

    const formData = new FormData();
    formData.append("pdf", pdfFile);
    formData.append("outputType", outputType);

    try {
      setLoading(true);
      setLoadingText("Reading your PDF...");
      setResult("Please wait while AI creates your notes.");
      setSourceImages([]);

      setTimeout(() => setLoadingText("Extracting important concepts..."), 1200);
      setTimeout(() => setLoadingText("Creating exam-ready output..."), 2500);
      setTimeout(
        () => setLoadingText("Large PDFs may take 1–3 minutes. Please wait..."),
        4000
      );

      const response = await fetch(
        "https://ai-notes-summarizer-vfus.onrender.com/upload",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        setResult("Backend Error: " + (data.error || "Unknown backend error"));
      } else if (data.error) {
        setResult("Error: " + data.error);
      } else {
        setResult(cleanMathOutput(data.text));
        setSourceImages(data.images || []);
      }
    } catch (error) {
      setResult(
        "Network/Backend Error: The backend did not respond properly. This may happen if Render is waking up, redeploying, or the PDF is too large. Please wait 1 minute and try again."
      );
    } finally {
      setLoading(false);
      setLoadingText("");
    }
  };

  const handleDownloadPDF = async () => {
    if (
      !result ||
      result === "Your AI-generated notes will appear here." ||
      result === "Please wait while AI creates your notes."
    ) {
      alert("Please generate notes first");
      return;
    }

    const element = resultRef.current;
    const isFormulaSheet = outputType === "Formula Sheet";

    const canvas = await html2canvas(element, {
      scale: isFormulaSheet ? 1.05 : 1.2,
      useCORS: true,
      allowTaint: true,
      backgroundColor: "#ffffff",
      windowWidth: element.scrollWidth,
      windowHeight: element.scrollHeight,
      onclone: (clonedDoc) => {
        const clonedElement = clonedDoc.querySelector(".pdf-content");

        if (clonedElement) {
          clonedElement.style.background = "#ffffff";
          clonedElement.style.color = "#1e293b";
          clonedElement.style.width = isFormulaSheet ? "1300px" : "1000px";
          clonedElement.style.borderRadius = "0";
          clonedElement.style.overflow = "visible";
        }
      },
    });

    const imgData = canvas.toDataURL("image/jpeg", 0.72);

    const pdf = new jsPDF({
      orientation: isFormulaSheet ? "l" : "p",
      unit: "mm",
      format: "a4",
      compress: true,
    });

    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();

    const imgWidth = pdfWidth;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;

    let heightLeft = imgHeight;
    let position = 0;

    pdf.addImage(
      imgData,
      "JPEG",
      0,
      position,
      imgWidth,
      imgHeight,
      undefined,
      "FAST"
    );

    heightLeft -= pdfHeight;

    while (heightLeft > 0) {
      position = heightLeft - imgHeight;
      pdf.addPage();

      pdf.addImage(
        imgData,
        "JPEG",
        0,
        position,
        imgWidth,
        imgHeight,
        undefined,
        "FAST"
      );

      heightLeft -= pdfHeight;
    }

    pdf.save(`ExamEase-${outputType}.pdf`);
  };

  const handleReset = () => {
    setPdfFile(null);
    setOutputType("Summary");
    setResult("Your AI-generated notes will appear here.");
    setSourceImages([]);
    setLoading(false);
    setLoadingText("");

    const fileInput = document.querySelector('input[type="file"]');

    if (fileInput) {
      fileInput.value = "";
    }
  };

  const InlineSourceImages = () => {
    if (outputType !== "Summary" || sourceImages.length === 0) {
      return null;
    }

    return (
      <div className="inline-source-image-group">
        {sourceImages.map((image, index) => (
          <div className="inline-source-image-wrapper" key={index}>
            <img
              src={image.src}
              alt={`Source PDF visual from page ${image.page}`}
              className="inline-source-image"
            />
            <p>Source PDF visual from page {image.page}</p>
          </div>
        ))}
      </div>
    );
  };

  const splitSummaryForImages = (text) => {
    if (outputType !== "Summary" || sourceImages.length === 0) {
      return {
        before: text,
        after: "",
      };
    }

    const markers = [
      "### 4. Detailed Explanation of Topics",
      "### 5. Important Formulas / Laws / Rules",
      "### 6. Important Tables",
    ];

    for (const marker of markers) {
      const index = text.indexOf(marker);

      if (index !== -1) {
        return {
          before: text.slice(0, index),
          after: text.slice(index),
        };
      }
    }

    return {
      before: text,
      after: "",
    };
  };

  const renderNormalOutput = () => {
    const parts = splitSummaryForImages(result);

    return (
      <div className="markdown-output">
        <ReactMarkdown
          remarkPlugins={markdownPlugins}
          rehypePlugins={rehypePlugins}
        >
          {parts.before}
        </ReactMarkdown>

        <InlineSourceImages />

        {parts.after && (
          <ReactMarkdown
            remarkPlugins={markdownPlugins}
            rehypePlugins={rehypePlugins}
          >
            {parts.after}
          </ReactMarkdown>
        )}
      </div>
    );
  };

  const isUsefulFormulaCard = (card) => {
    const cleaned = card
      .replace(/^## CARD.*$/gm, "")
      .replace(/\*\*Diagram:\*\*/g, "")
      .replace(/\*\*Formula \/ Rule:\*\*/g, "")
      .replace(/\*\*Important Points:\*\*/g, "")
      .replace(/\*\*Exam Use:\*\*/g, "")
      .replace(/\*\*Note:\*\*/g, "")
      .replace(/[-•*\s:]/g, "");

    return cleaned.length > 70 && (card.includes("$") || card.includes("="));
  };

  const renderFormulaSheetOutput = () => {
    const quickSplit = result.split(
      /(?=^## QUICK FORMULAS|^## QUICK LAWS|^## QUICK FORMULAS \/ LAWS)/m
    );

    const mainFormulaText = quickSplit[0] || "";
    const quickSection = quickSplit.slice(1).join("\n\n");

    const parts = mainFormulaText.split(/(?=^## CARD\s*\d*[:.-])/m);
    const intro = parts[0] || "";
    const cards = parts.slice(1).filter(isUsefulFormulaCard);

    return (
      <div className="formula-sheet-output">
        <div className="formula-sheet-title">
          <ReactMarkdown
            remarkPlugins={markdownPlugins}
            rehypePlugins={rehypePlugins}
          >
            {intro}
          </ReactMarkdown>
        </div>

        <div className="formula-card-masonry">
          {cards.map((card, index) => (
            <div
              className={`formula-card card-color-${(index % 6) + 1}`}
              key={index}
            >
              <ReactMarkdown
                remarkPlugins={markdownPlugins}
                rehypePlugins={rehypePlugins}
              >
                {card}
              </ReactMarkdown>
            </div>
          ))}
        </div>

        {quickSection && (
          <div className="quick-formula-strip">
            <ReactMarkdown
              remarkPlugins={markdownPlugins}
              rehypePlugins={rehypePlugins}
            >
              {quickSection}
            </ReactMarkdown>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app">
      <div className="hero">
        <div className="badge">AI Powered Study Assistant</div>

        <h1>ExamEase AI</h1>

        <p className="subtitle">
          Convert your college PDFs into exam-ready summaries, formulas,
          important questions and viva notes.
        </p>
      </div>

      <div className="main-card">
        <div className="input-section">
          <div className="input-group">
            <label>Upload Your PDF</label>

            <input
              type="file"
              accept="application/pdf"
              onChange={handleFileChange}
            />

            {pdfFile && <p className="file-name">Selected: {pdfFile.name}</p>}
          </div>

          <div className="input-group">
            <label>Select Output Type</label>

            <select
              value={outputType}
              onChange={(e) => setOutputType(e.target.value)}
            >
              <option>Summary</option>
              <option>Important Questions</option>
              <option>MCQs</option>
              <option>Formula Sheet</option>
              <option>Viva Questions</option>
            </select>
          </div>

          <div className="button-row">
            <button
              className="generate-btn"
              onClick={handleGenerate}
              disabled={loading}
            >
              {loading ? "Generating..." : "Generate Notes"}
            </button>

            <button className="download-btn" onClick={handleDownloadPDF}>
              Download PDF
            </button>

            <button className="reset-btn" onClick={handleReset}>
              Reset
            </button>
          </div>

          {loading && (
            <div className="loading-box">
              <div className="loader"></div>
              <p>{loadingText}</p>
            </div>
          )}
        </div>
      </div>

      <div
        className={`result-box ${
          outputType === "Formula Sheet" ? "formula-result-box" : ""
        }`}
      >
        <div
          className={`pdf-content ${
            outputType === "Formula Sheet" ? "formula-sheet-mode" : ""
          }`}
          ref={resultRef}
        >
          <div className="pdf-header">
            <h2>ExamEase AI Notes</h2>
            <p>{outputType} · Generated from your uploaded PDF</p>
          </div>

          {outputType === "Formula Sheet"
            ? renderFormulaSheetOutput()
            : renderNormalOutput()}
        </div>
      </div>
    </div>
  );
}

export default App;