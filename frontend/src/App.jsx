import React, { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

const SUBJECTS = [
  "Anatomy",
  "Physiology",
  "Biochemistry",
  "Pathology",
  "Pharmacology",
  "Microbiology",
  "Forensic Medicine & Toxicology",
  "Community Medicine",
  "General Medicine",
  "General Surgery",
  "Pediatrics",
  "Obstetrics & Gynecology",
  "Orthopedics",
  "Dermatology",
  "Psychiatry",
  "Radiology",
  "Anesthesiology",
  "Ophthalmology",
  "Otorhinolaryngology (ENT)",
  "Dentistry / Oral Medicine",
  "Cardiology",
  "Neurology",
  "Pulmonology",
  "Gastroenterology",
  "Nephrology",
  "Endocrinology",
  "Rheumatology",
  "Oncology",
  "Urology",
  "Emergency Medicine",
];

const SUGGESTED_QUESTIONS = {
  Anatomy: [
    "Explain the structure of the brain",
    "What are the parts of the heart?",
    "Explain the anatomy of the kidney",
  ],
  Physiology: [
    "Explain the cardiac cycle",
    "How does respiration work?",
    "Explain renal filtration",
  ],
  Cardiology: [
    "What is heart failure?",
    "Explain the cardiac cycle",
    "What are the chambers of the heart?",
  ],
  Neurology: [
    "Explain the brainstem",
    "What is the function of the cerebellum?",
    "Explain the cranial nerves",
  ],
  Pulmonology: [
    "Explain the anatomy of the lungs",
    "How does gas exchange occur?",
    "Explain the pulmonary circulation",
  ],
  Pathology: [
    "What is inflammation?",
    "Explain cellular injury",
    "What is necrosis?",
  ],
  Pharmacology: [
    "What is pharmacokinetics?",
    "Explain drug absorption",
    "What is the mechanism of drug action?",
  ],
  Ophthalmology: [
    "Explain the anatomy of the eye",
    "What are the layers of the eye?",
    "Explain the function of the retina",
  ],
  "Otorhinolaryngology (ENT)": [
    "Explain the anatomy of the ear",
    "What are the parts of the middle ear?",
    "Explain the anatomy of the nose",
  ],
};

export default function App() {
  const [subject, setSubject] = useState("Cardiology");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [imageResult, setImageResult] = useState(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState("");

  const imageInputRef = useRef(null);

  function scrollToSection(id) {
    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  async function askQuestion(customQuestion = null) {
    const finalQuestion =
      customQuestion !== null ? customQuestion : question;

    if (!finalQuestion.trim()) {
      setError("Please enter a medical question.");
      return;
    }

    setQuestion(finalQuestion);
    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject,
          question: finalQuestion,
        }),
      });

      const data = await response.json();

      if (!response.ok || data.success === false) {
        throw new Error(
          data.detail || data.error || "Unable to get an answer."
        );
      }

      setAnswer(data.answer || "");
      setSources(Array.isArray(data.sources) ? data.sources : []);
    } catch (err) {
      console.error("ASK QUESTION ERROR:", err);
      setError(
        err.message ||
        "Something went wrong while contacting the backend."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleImageSelect(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    const allowedTypes = ["image/png", "image/jpeg", "image/webp"];

    if (!allowedTypes.includes(file.type)) {
      setImageError("Please select a PNG, JPG, JPEG or WEBP image.");
      event.target.value = "";
      return;
    }

    if (imagePreview) URL.revokeObjectURL(imagePreview);

    setSelectedImage(file);
    setImagePreview(URL.createObjectURL(file));
    setImageError("");
    setImageResult(null);
  }

  function clearImage() {
    if (imagePreview) URL.revokeObjectURL(imagePreview);

    setSelectedImage(null);
    setImagePreview("");
    setImageResult(null);
    setImageError("");

    if (imageInputRef.current) {
      imageInputRef.current.value = "";
    }
  }

  async function analyzeImage() {
    if (!selectedImage) {
      setImageError("Please select a medical image first.");
      return;
    }

    setImageLoading(true);
    setImageError("");
    setImageResult(null);

    try {
      const formData = new FormData();
      formData.append("image", selectedImage);

      const response = await fetch(
        `${API_BASE_URL}/api/analyze-image`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok || data.success === false) {
        throw new Error(
          data.detail || data.error || "Image analysis failed."
        );
      }

      setImageResult(data);
    } catch (err) {
      console.error("IMAGE ANALYSIS ERROR:", err);
      setImageError(err.message || "Unable to analyze the image.");
    } finally {
      setImageLoading(false);
    }
  }

  const suggestedQuestions =
    SUGGESTED_QUESTIONS[subject] || SUGGESTED_QUESTIONS.Anatomy;

  return (
    <div className="medical-app">
      <header className="header">
        <div className="brand">
          <div className="brand-icon">🧬</div>
          <div>
            <div className="brand-name">MedicalAI</div>
            <div className="brand-subtitle">
              AI-Powered Medical Learning
            </div>
          </div>
        </div>

        <nav className="main-nav">
          <button onClick={() => scrollToSection("home")}>◇ Home</button>
          <button onClick={() => scrollToSection("ask")}>✦ Ask AI</button>
          <button onClick={() => scrollToSection("image-ai")}>◉ Image AI</button>
          <button onClick={() => scrollToSection("topics")}>▤ Topics</button>
          <button onClick={() => scrollToSection("about")}>ⓘ About</button>
        </nav>

        <div className="profile-circle">A</div>
      </header>

      <section id="home" className="hero">
        <div className="hero-badge">
          ✦ AI-POWERED MEDICAL EDUCATION
        </div>

        <h1>
          Learn Medicine<span> Smarter.</span>
        </h1>

        <p>
          Ask medical questions, analyze medical images and
          understand complex medical concepts with AI.
        </p>

        <div className="hero-buttons">
          <button
            className="hero-primary"
            onClick={() => scrollToSection("ask")}
          >
            ✦ Ask Medical AI
          </button>
          <button
            className="hero-secondary"
            onClick={() => scrollToSection("image-ai")}
          >
            ◉ Analyze Image
          </button>
        </div>
      </section>

      <main className="main-container">
        <section id="ask" className="dashboard-grid">
          <div className="question-panel">
            <div className="panel-heading">
              <div className="heading-icon">✦</div>
              <div>
                <h2>Ask a Medical Question</h2>
                <p>Get AI-powered explanations</p>
              </div>
            </div>

            <label>Medical Subject</label>
            <select
              value={subject}
              onChange={(event) => setSubject(event.target.value)}
            >
              {SUBJECTS.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>

            <label>Your Question</label>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask anything about medicine..."
              rows={5}
            />

            <button
              className="primary-button"
              onClick={() => askQuestion()}
              disabled={loading}
            >
              {loading ? "✦ Generating Answer..." : "✦ Ask Medical AI"}
            </button>

            <div className="suggested-heading">Suggested Questions</div>
            <div className="suggested-list">
              {suggestedQuestions.map((item) => (
                <button key={item} onClick={() => askQuestion(item)}>
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="answer-panel">
            <div className="panel-heading">
              <div className="heading-icon purple">✦</div>
              <div>
                <h2>AI Answer</h2>
                <p>RAG-powered medical explanation</p>
              </div>
            </div>

            {error && <div className="error-box">⚠ {error}</div>}

            {loading ? (
              <div className="loading-state">
                <div className="spinner" />
                <h3>Searching medical knowledge...</h3>
                <p>
                  Retrieving trusted information and generating your
                  explanation.
                </p>
              </div>
            ) : answer ? (
              <div className="answer-content">
                <div className="verified-message">
                  ✓ AI explanation generated using retrieved medical
                  information.
                </div>
                <ReactMarkdown>{answer}</ReactMarkdown>
              </div>
            ) : (
              <div className="empty-answer">
                <div className="empty-icon">✦</div>
                <h3>Your medical answer will appear here</h3>
                <p>Ask a question to begin.</p>
              </div>
            )}
          </div>
        </section>

        <section id="image-ai" className="image-ai-section">
          <div className="section-top">
            <div>
              <div className="section-label">✦ IMAGE UNDERSTANDING</div>
              <h2>Medical Image AI</h2>
              <p>
                Upload a medical image and let MedicalAI analyze it
                automatically.
              </p>
            </div>
            <div className="auto-badge">AUTOMATIC AI ANALYSIS</div>
          </div>

          <div className="image-analysis-grid">
            <div className="upload-card">
              {!imagePreview ? (
                <label className="upload-box">
                  <input
                    ref={imageInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/jpg,image/webp"
                    onChange={handleImageSelect}
                  />
                  <div className="upload-icon">↑</div>
                  <h3>Upload Medical Image</h3>
                  <p>PNG, JPG, JPEG or WEBP</p>
                  <span className="choose-button">Choose Image</span>
                </label>
              ) : (
                <div className="selected-image-box">
                  <img src={imagePreview} alt="Selected medical image" />
                  <button className="change-image" onClick={clearImage}>
                    Change Image
                  </button>
                </div>
              )}

              {imageError && (
                <div className="image-error">⚠ {imageError}</div>
              )}

              <button
                className="analyze-button"
                onClick={analyzeImage}
                disabled={!selectedImage || imageLoading}
              >
                {imageLoading ? "✦ Analyzing Image..." : "✦ Analyze Image"}
              </button>
            </div>

            <div className="image-result-card">
              {!imageResult ? (
                <div className="result-empty">
                  <div className="result-empty-icon">◉</div>
                  <h3>AI Analysis Result</h3>
                  <p>
                    Upload an image to identify structures, determine
                    the medical subject and retrieve educational
                    information.
                  </p>
                  <div className="pipeline">
                    <span>Image</span>
                    <b>→</b>
                    <span>Vision AI</span>
                    <b>→</b>
                    <span>RAG</span>
                  </div>
                </div>
              ) : (
                <div>
                  <div className="result-header">
                    <div>
                      <div className="result-label">
                        IDENTIFIED STRUCTURE
                      </div>
                      <h2>
                        {imageResult.identified_structure ||
                          "Unknown Structure"}
                      </h2>
                    </div>
                    <div className="confidence-circle">
                      {Math.round(
                        Number(imageResult.vision_confidence || 0)
                      )}%
                    </div>
                  </div>

                  <div className="analysis-details">
                    <div>
                      <span>Image Type</span>
                      <strong>{imageResult.image_type || "Unknown"}</strong>
                    </div>
                    <div>
                      <span>Modality</span>
                      <strong>{imageResult.modality || "Unknown"}</strong>
                    </div>
                    <div>
                      <span>Body Region</span>
                      <strong>{imageResult.body_region || "Unknown"}</strong>
                    </div>
                    <div>
                      <span>Medical Subject</span>
                      <strong>
                        {imageResult.medical_subject || "Anatomy"}
                      </strong>
                    </div>
                  </div>

                  <div className="medical-warning">
                    ⚠ Educational Use Only — This AI interpretation is
                    for medical education and should not be treated as
                    a clinical diagnosis.
                  </div>

                  {imageResult.description && (
                    <div className="explanation-card">
                      <div className="explanation-title">
                        ◉ Image Description
                      </div>
                      <p>{imageResult.description}</p>
                    </div>
                  )}

                  {imageResult.gemini?.answer && (
                    <div className="explanation-card">
                      <div className="explanation-title">
                        ✦ AI Educational Explanation
                      </div>
                      <div className="markdown-content">
                        <ReactMarkdown>
                          {imageResult.gemini.answer}
                        </ReactMarkdown>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        {sources.length > 0 && (
          <section className="sources-section">
            <div className="section-heading">
              <div>
                <span>TRUSTED KNOWLEDGE</span>
                <h2>Retrieved Sources</h2>
              </div>
            </div>

            <div className="sources-grid">
              {sources.map((source, index) => (
                <div
                  className="source-card"
                  key={`${source.source || "source"}-${index}`}
                >
                  <div className="source-icon">✓</div>
                  <div>
                    <strong>{source.source || "Medical Source"}</strong>
                    <p>
                      {source.topic ||
                        source.subtopic ||
                        "Medical reference"}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <section id="topics" className="topics-section">
          <div className="section-heading">
            <div>
              <span>EXPLORE MEDICINE</span>
              <h2>30 Medical Subjects</h2>
            </div>
          </div>

          <div className="topics-grid">
            {SUBJECTS.map((item, index) => (
              <button
                className="topic-card"
                key={item}
                onClick={() => {
                  setSubject(item);
                  scrollToSection("ask");
                }}
              >
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{item}</strong>
                <b>→</b>
              </button>
            ))}
          </div>
        </section>

        <section id="about" className="about-section">
          <div className="about-card">
            <div className="about-icon">🧬</div>
            <div>
              <div className="section-label">ABOUT MEDICAL AI</div>
              <h2>AI + Medical Knowledge</h2>
              <p>
                MedicalAI combines Retrieval-Augmented Generation,
                medical knowledge retrieval and AI image understanding
                to create an educational learning environment for
                medical students.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <div className="footer-brand">
          <div className="brand-icon">🧬</div>
          <div>
            <strong>MedicalAI</strong>
            <span>AI-powered medical learning</span>
          </div>
        </div>
        <div className="footer-right">
          <span>Educational Use Only</span>
          <span>© 2026 MedicalAI</span>
        </div>
      </footer>
    </div>
  );
}