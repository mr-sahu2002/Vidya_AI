import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Persona.css";

const questions = [
  {
    id: 1,
    text: "When your teacher explains something new, what helps you understand best?",
    options: {
      A: "Step-by-step explanations with clear examples",
      B: "Stories and analogies that relate to things I know",
      C: "Simple, direct explanations without extra details",
      D: "Detailed explanations that cover everything thoroughly",
    },
  },
  {
    id: 2,
    text: "If you had to learn about animals, which would you prefer?",
    options: {
      A: "Reading interesting facts and stories about animals",
      B: 'Learning through comparisons ("A lion is like a big house cat")',
      C: "Short, simple descriptions of each animal",
      D: "Detailed explanations about how animals live and behave",
    },
  },
  

  {
    id: 3,
    text: "When you're trying to remember something important, what works best?",
    options: {
      A: "Breaking it into smaller, simpler parts",
      B: "Connecting it to stories or things I already know",
      C: "Repeating the main points several times",
      D: "Understanding all the details and reasons why",
    },
  },
  {
    id: 4,
    text: "When you get stuck on a difficult question, what do you usually do first?",
    options: {
      A: "Look for similar examples to help me understand",
      B: "Ask someone to explain it using simpler words",
      C: "Try different approaches until something works",
      D: "Break it into smaller, easier steps",
    },
  },
  {
    id: 5,
    text: "Which type of learning activity would you choose?",
    options: {
      A: "Word games and vocabulary puzzles",
      B: "Story-based learning with characters and plots",
      C: "Hands-on practice problems to solve",
      D: "Step-by-step tutorials that build up knowledge",
    },
  },
  {
    id: 6,
    text: "When working on group projects, you prefer to:",
    options: {
      A: "Be the leader who organizes and guides everyone",
      B: "Share ideas and discuss solutions with others",
      C: "Focus on completing specific tasks assigned to you",
      D: "Research and gather information for the group",
    },
  },
  {
    id: 7,
    text: "If you don't understand something in class, you:",
    options: {
      A: "Raise your hand immediately to ask questions",
      B: "Wait and ask the teacher privately after class",
      C: "Ask a friend to help explain it to you",
      D: "Try to figure it out yourself using notes or books",
    },
  },
  {
    id: 8,
    text: "What makes you most excited to learn something new?",
    options: {
      A: "When it's about something I'm curious about",
      B: "When I can use it to help others",
      C: "When it's challenging and makes me think hard",
      D: "When I can see how it connects to my hobbies",
    },
  },
  {
    id: 9,
    text: "Which reward motivates you most when you do well?",
    options: {
      A: "Praise from teacher in front of everyone",
      B: "A quiet 'well done' just to me",
      C: "Getting to help teach other students",
      D: "Having free time to explore what interests me",
    },
  },
  {
    id: 10,
    text: "When do you focus best during lessons?",
    options: {
      A: "When I can ask questions and interact with the teacher",
      B: "When explanations are given clearly and slowly",
      C: "When I can work through problems myself",
      D: "When the material builds logically from simple to complex",
    },
  },
  {
    id: 11,
    text: "What type of explanation helps you learn best?",
    options: {
      A: "Short, direct answers that get to the point quickly",
      B: "Detailed explanations with lots of examples",
      C: "Explanations that relate to my interests and hobbies",
      D: "Step-by-step explanations that don't skip any steps",
    },
  },
  {
    id: 12,
    text: "When you make a mistake, how do you usually feel?",
    options: {
      A: "Frustrated but ready to try again quickly",
      B: "A little upset but I learn from it",
      C: "I want to understand exactly what went wrong",
      D: "I prefer to ask for help before trying again",
    },
  },
  {
    id: 13,
    text: "Which describes how you like to practice new skills?",
    options: {
      A: "Practice a lot until I get it perfect",
      B: "Practice a little bit every day",
      C: "Practice with friends so it's more fun",
      D: "Practice only when I feel like it",
    },
  },
  {
    id: 14,
    text: "Which school subject do you find most interesting?",
    options: {
      A: "Art, Design, or subjects with creative projects",
      B: "Language, Literature, or subjects with stories",
      C: "Science, Math, or subjects with logical thinking",
      D: "History, Social Studies, or subjects about people",
    },
  },
  {
    id: 15,
    text: "When reading a story, what interests you most?",
    options: {
      A: "Imagining what the characters and places look like",
      B: "Understanding what the characters are feeling",
      C: "Following the plot and predicting what happens next",
      D: "Learning new facts or information from the story",
    },
  },
  {
    id: 16,
    text: "When learning new topics, you prefer:",
    options: {
      A: "Interactive discussions and Q&A sessions",
      B: "Reading materials with clear explanations",
      C: "Practice exercises and problem-solving activities",
      D: "Comprehensive lessons that cover all details thoroughly",
    },
  },
  {
    id: 17,
    text: "When something seems too difficult, you:",
    options: {
      A: "Ask for an easier version to start with",
      B: "Want someone to walk me through it step by step",
      C: "Keep trying different approaches until it works",
      D: "Take a break and come back to it later",
    },
  },
  {
    id: 18,
    text: "Your ideal way to show what you've learned is:",
    options: {
      A: "Explaining it in your own words out loud or in writing",
      B: "Answering detailed questions about the topic",
      C: "Solving problems that test your understanding",
      D: "Writing a comprehensive summary of everything learned",
    },
  },
  {
    id: 19,
    text: "When learning something new, you prefer to:",
    options: {
      A: "Learn it quickly with key points and move on",
      B: "Take time to understand it deeply with examples",
      C: "Learn through practice and repeated exercises",
      D: "Study it thoroughly until you master every detail",
    },
  },
  
  {
    id: 20,
    text: "Which describes how you like to receive feedback?",
    options: {
      A: "Quick, encouraging comments that keep me motivated",
      B: "Detailed feedback that explains what I did well and what to improve",
      C: "Practical suggestions for how to do better next time",
      D: "Comprehensive reviews that analyze my work thoroughly",
    },
  },
];

// Sidebar items, same as Whiteboard page
const SIDEBAR_ITEMS = [
  { key: "whiteboard", label: "Whiteboard" },
  { key: "assignments", label: "Assignments" },
  { key: "questionnaire", label: "Questionnaire" }, // active page
  { key: "games", label: "Games" },
  { key: "chat", label: "Chat" },
  { key: "notes", label: "Notes" },
];

// Mock user info
const user = {
  name: "Sarah Lee",
  email: "sarah.lee@example.com",
  usn: "2CS22CS123",
};

export default function Persona() {
  const navigate = useNavigate();
  const [answers, setAnswers] = useState({});
  const [showProfile, setShowProfile] = useState(false);
  const [error, setError] = useState(false);

  const handleOptionChange = (questionId, selectedOption) => {
    setAnswers((prev) => ({
      ...prev,
      [questionId]: selectedOption,
    }));
  };

  const handleSubmit = () => {
    if (Object.keys(answers).length !== questions.length) {
      setError(true);
    } else {
      setError(false);
      // Save answers to backend if needed before navigation
      navigate("/login");
    }
  };

  return (
    <div className="personaDashboard">
      {/* Sidebar */}
      <aside className="personaSidebar">
        <div className="sidebarLogo">
          <span className="emoji">📚</span>
          <span className="brand">VidyaAI</span>
        </div>
      </aside>

      {/* Main Content */}
      <main className="personaMain">
        {/* Header with profile icon */}
        <header className="personaHeader">
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
        </header>

        {/* Questionnaire content card */}
        <section className="personaContent">
          <h2 className="personaTitle">Learning Persona</h2>
          <form className="personaForm" onSubmit={(e) => e.preventDefault()}>
            {questions.map((q) => (
              <div key={q.id} className="personaQuestion">
                <p className="personaQuestionText">
                  {q.id}. {q.text}
                </p>
                <div className="personaOptionsGroup">
                  {Object.entries(q.options).map(([key, label]) => (
                    <label key={key} className="personaOption">
                      <input
                        type="radio"
                        name={`question-${q.id}`}
                        value={key}
                        checked={answers[q.id] === key}
                        onChange={() => handleOptionChange(q.id, key)}
                      />
                      {key}) {label}
                    </label>
                  ))}
                </div>
              </div>
            ))}
            <button
              type="button"
              className="personaSubmitBtn"
              onClick={handleSubmit}
            >
              Submit
            </button>
            {error && (
              <div className="personaError">
                Looks like you haven't answered all the questions. Please fill
                all the answers.
              </div>
            )}
          </form>
        </section>
      </main>
    </div>
  );
}
