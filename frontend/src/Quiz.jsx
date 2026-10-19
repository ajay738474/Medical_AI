import { useState } from "react";
import "./Quiz.css";

const TOPICS = {
    Anatomy: [
        "Four Chambers of the Heart",
        "Heart Valves",
        "Blood Circulation",
        "Human Skeletal System",
        "Brain Structure",
    ],
    Physiology: [
        "Cardiac Cycle",
        "Blood Pressure",
        "Respiratory System",
        "Nerve Impulses",
    ],
    Biochemistry: [
        "Carbohydrate Metabolism",
        "Proteins and Enzymes",
        "ATP Production",
    ],
    Histology: [
        "Cardiac Muscle Tissue",
        "Epithelial Tissue",
        "Connective Tissue",
    ],
    Pathology: [
        "Myocardial Infarction",
        "Inflammation",
        "Anemia",
    ],
    Pharmacology: [
        "Beta Blockers",
        "Antibiotics",
        "Drug Absorption",
    ],
};

const API_BASE_URL = "http://127.0.0.1:8000";

export default function Quiz() {
    const [subject, setSubject] = useState("Anatomy");
    const [topic, setTopic] = useState(
        "Four Chambers of the Heart"
    );
    const [questions, setQuestions] = useState([]);
    const [current, setCurrent] = useState(0);
    const [selected, setSelected] = useState(null);
    const [answers, setAnswers] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [finished, setFinished] = useState(false);

    const topics = TOPICS[subject] || [];
    const question = questions[current];

    function changeSubject(value) {
        setSubject(value);
        setTopic(TOPICS[value]?.[0] || "");
        setQuestions([]);
        setCurrent(0);
        setSelected(null);
        setAnswers([]);
        setFinished(false);
        setError("");
    }

    async function startQuiz() {
        setLoading(true);
        setError("");
        setQuestions([]);
        setAnswers([]);
        setCurrent(0);
        setSelected(null);
        setFinished(false);

        try {
            const response = await fetch(`${API_BASE_URL}/api/quiz`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    subject,
                    topic,
                    count: 20,
                    difficulty: "mixed",
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Could not generate quiz.");
            }

            const generated = data.questions;

            if (!Array.isArray(generated) || generated.length !== 20) {
                throw new Error("The server must return exactly 20 questions.");
            }

            for (const item of generated) {
                if (
                    typeof item.question !== "string" ||
                    !Array.isArray(item.options) ||
                    item.options.length !== 4 ||
                    !Number.isInteger(item.answer) ||
                    item.answer < 0 ||
                    item.answer > 3
                ) {
                    throw new Error("The server returned an invalid question.");
                }
            }

            setQuestions(generated);
        } catch (err) {
            setError(
                `${err.message} Check that your backend is running and /api/quiz is configured.`
            );
        } finally {
            setLoading(false);
        }
    }

    function submitAnswer() {
        if (selected === null || !question) return;

        setAnswers((previous) => [
            ...previous,
            {
                question: question.question,
                chosen: selected,
                correct: question.answer,
                explanation: question.explanation || "",
            },
        ]);

        if (current === questions.length - 1) {
            setFinished(true);
        } else {
            setCurrent((previous) => previous + 1);
        }

        setSelected(null);
    }

    function restartQuiz() {
        setQuestions([]);
        setCurrent(0);
        setSelected(null);
        setAnswers([]);
        setFinished(false);
        setError("");
    }

    const score = answers.filter(
        (item) => item.chosen === item.correct
    ).length;

    return (
        <section className="topic-quiz" id="topic-quiz">
            <div className="quiz-heading">
                <span className="quiz-eyebrow">MEDICALAI LEARNING</span>
                <h2>Topic-Based AI Quiz</h2>
                <p>
                    Study a medical topic and test your knowledge with
                    20 AI-generated questions.
                </p>
            </div>

            {questions.length === 0 && !loading && (
                <div className="quiz-setup">
                    <label htmlFor="quiz-subject">Medical subject</label>
                    <select
                        id="quiz-subject"
                        value={subject}
                        onChange={(e) => changeSubject(e.target.value)}
                    >
                        {Object.keys(TOPICS).map((name) => (
                            <option key={name} value={name}>
                                {name}
                            </option>
                        ))}
                    </select>

                    <label htmlFor="quiz-topic">Choose a specific topic</label>
                    <select
                        id="quiz-topic"
                        value={topic}
                        onChange={(e) => setTopic(e.target.value)}
                    >
                        {topics.map((name) => (
                            <option key={name} value={name}>
                                {name}
                            </option>
                        ))}
                    </select>

                    <div className="quiz-topic-note">
                        <strong>Your quiz:</strong> 20 questions about{" "}
                        {topic || "your selected topic"}.
                    </div>

                    <button
                        className="quiz-primary"
                        onClick={startQuiz}
                        disabled={!topic}
                    >
                        Generate AI Quiz
                    </button>

                    {error && <p className="quiz-error">{error}</p>}
                </div>
            )}

            {loading && (
                <div className="quiz-loading" role="status">
                    <div className="quiz-spinner" />
                    <h3>Generating your quiz...</h3>
                    <p>
                        Creating questions about {topic}. This may take a
                        few moments.
                    </p>
                </div>
            )}

            {question && !finished && (
                <div className="quiz-question-card">
                    <div className="quiz-progress-text">
                        <span>{subject}</span>
                        <span>
                            Question {current + 1} of {questions.length}
                        </span>
                    </div>

                    <div className="quiz-progress">
                        <div
                            style={{
                                width: `${((current + 1) / questions.length) * 100}%`,
                            }}
                        />
                    </div>

                    <p className="quiz-topic-label">{topic}</p>
                    <h3>{question.question}</h3>

                    <div className="quiz-options">
                        {question.options.map((option, index) => (
                            <button
                                key={`${index}-${option}`}
                                className={`quiz-option ${selected === index ? "selected" : ""
                                    }`}
                                onClick={() => setSelected(index)}
                            >
                                <span>{String.fromCharCode(65 + index)}</span>
                                {option}
                            </button>
                        ))}
                    </div>

                    <button
                        className="quiz-primary"
                        onClick={submitAnswer}
                        disabled={selected === null}
                    >
                        {current === questions.length - 1
                            ? "Finish Quiz"
                            : "Next Question"}
                    </button>
                </div>
            )}

            {finished && (
                <div className="quiz-results">
                    <h3>Quiz Completed!</h3>
                    <p className="quiz-score">
                        {score} / {questions.length}
                    </p>
                    <p>
                        You scored{" "}
                        {Math.round((score / questions.length) * 100)}%
                        on {topic}.
                    </p>

                    <div className="quiz-review">
                        <h4>Review your answers</h4>
                        {answers.map((item, index) => (
                            <div className="quiz-review-item" key={index}>
                                <strong>
                                    {index + 1}. {item.question}
                                </strong>
                                <p>
                                    Your answer:{" "}
                                    {item.chosen === item.correct
                                        ? "Correct"
                                        : item.chosen >= 0
                                            ? "Incorrect"
                                            : "Not answered"}
                                </p>
                                <p>
                                    Correct answer: {item.correct + 1}.{" "}
                                    {questions[index]?.options[item.correct]}
                                </p>
                                {item.explanation && <p>{item.explanation}</p>}
                            </div>
                        ))}
                    </div>

                    <button className="quiz-primary" onClick={restartQuiz}>
                        Choose Another Topic
                    </button>
                </div>
            )}
        </section>
    );
}