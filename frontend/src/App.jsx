import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import "katex/dist/katex.min.css";
import "./App.css";

function App() {
  const [pdfFile, setPdfFile] = useState(null);
  const [outputType, setOutputType] = useState("Summary");
  const [result, setResult] = useState("Your AI-generated notes will appear here.");
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("");

  const resultRef = useRef(null);

  const handleFileChange = (e) => {
    setPdfFile(e.target.files[0]);
  };

  const cleanMathOutput = (text) => {
    return text
      .replace(/\\\[/g, "$$")
      .replace(/\\\]/g, "$$")
      .replace(/\\\(/g, "$")
      .replace(/\\\)/g, "$");
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

      setTimeout(() => setLoadingText("Extracting important concepts..."), 1200);
      setTimeout(() => setLoadingText("Creating exam-ready notes..."), 2500);
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

      if (data.error) {
        setResult("Error: " + data.error);
      } else {
        setResult(cleanMathOutput(data.text));
      }
    } catch (error) {
      setResult("Something went wrong. Make sure backend is running.");
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

    const canvas = await html2canvas(element, {
      scale: 2,
      useCORS: true,
      backgroundColor: "#ffffff",
    });

    const imgData = canvas.toDataURL("image/png");

    const pdf = new jsPDF("p", "mm", "a4");

    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();

    const imgWidth = pdfWidth;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;

    let heightLeft = imgHeight;
    let position = 0;

    pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);
    heightLeft -= pdfHeight;

    while (heightLeft > 0) {
      position = heightLeft - imgHeight;
      pdf.addPage();
      pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);
      heightLeft -= pdfHeight;
    }

    pdf.save(`ExamEase-${outputType}.pdf`);
  };

  const handleReset = () => {
    setPdfFile(null);
    setOutputType("Summary");
    setResult("Your AI-generated notes will appear here.");
    setLoading(false);
    setLoadingText("");

    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
      fileInput.value = "";
    }
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

      <div className="result-box">
        <div className="pdf-content" ref={resultRef}>
          <h2>Generated Output</h2>

          <div className="markdown-output">
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {result}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;