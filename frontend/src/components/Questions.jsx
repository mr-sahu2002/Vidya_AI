import React, { useState, useEffect } from "react";
import "./Questions.css";

const SIDEBAR_ITEMS = [
  { key: "whiteboard", label: "Whiteboard" },
  { key: "assignments", label: "Assignments" },
  { key: "questionnaire", label: "Questionnaire" },
  { key: "games", label: "Games" },
  { key: "chat", label: "Chat" },
  { key: "notes", label: "Notes" },
];

// Mock user profile
const user = {
  name: "Sarah Lee",
  email: "sarah.lee@example.com",
  usn: "2CS22CS123",
};

// Mock backend questions (replace with your backend API call)
const mockFetchQuestions = () =>
  Promise.resolve([
    { id: 1, question: "What is the capital of France?" },
    { id: 2, question: "Explain the concept of photosynthesis." },
    { id: 3, question: "What are Newton's three laws of motion?" },
  ]);

export default function Questions() {
  const [showProfile, setShowProfile] = useState(false);
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});

  useEffect(() => {
    // Replace with actual fetch from backend
    mockFetchQuestions().then((data) => setQuestions(data));
  }, []);

  function handleAnswerChange(questionId, value) {
    setAnswers((prev) => ({
      ...prev,
      [questionId]: value,
    }));
  }

  function handleSubmit(e) {
    e.preventDefault();
    // Here you can send `answers` to backend as needed
    console.log("Submitted answers:", answers);
    alert("Answers submitted successfully!");
  }

  return (
    <div className="questionsDashboard">
      {/* Sidebar */}
      <aside className="questionsSidebar">
        <div className="sidebarLogo">
          <span className="emoji">📚</span>
          <span className="brand">VidyaAI</span>
        </div>
        <nav className="sidebarNav">
          {SIDEBAR_ITEMS.map((item) => (
            <button
              key={item.key}
              className={`sidebarBtn${item.key === "questionnaire" ? " active" : ""}`}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      {/* Main Section */}
      <main className="questionsMain">
        {/* Top right profile */}
        <div className="questionsHeader">
          <button
            className="profileAvatar"
            title="Profile"
            onClick={() => setShowProfile((v) => !v)}
          >
            {user.email.charAt(0).toUpperCase()}
          </button>
          {showProfile && (
            <div className="profilePopover" onClick={() => setShowProfile(false)}>
              <div className="profileInfoName">{user.name}</div>
              <div className="profileInfoEmail">{user.email}</div>
              <div className="profileInfoUSN">USN: {user.usn}</div>
            </div>
          )}
        </div>

        {/* Questions Content */}
        <section className="questionsContent">
          <h2 className="questionsTitle">Questions</h2>
          <form onSubmit={handleSubmit} className="questionsForm" noValidate>
            {questions.length === 0 && <p>No questions available.</p>}
            {questions.map(({ id, question }) => (
              <div key={id} className="questionItem">
                <label htmlFor={`answer-${id}`} className="questionText">
                  {question}
                </label>
                <textarea
                  id={`answer-${id}`}
                  className="answerInput"
                  placeholder="Type your answer here..."
                  value={answers[id] || ""}
                  onChange={(e) => handleAnswerChange(id, e.target.value)}
                  rows={4}
                  required
                />
              </div>
            ))}
            {questions.length > 0 && (
              <button type="submit" className="submitAnswersBtn">
                Submit Answers
              </button>
            )}
          </form>
        </section>
      </main>
    </div>
  );
}
