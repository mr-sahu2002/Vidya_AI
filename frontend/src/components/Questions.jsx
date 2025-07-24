import React from "react";
import { useNavigate } from "react-router-dom";
const sidebarButtons = [
  { label: "Video", path: "/student/video" },
  { label: "Notes", path: "/student/notes" },
  { label: "Games", path: "/student/games" },
  { label: "Assignment", path: "/student/assignment" },
  { label: "Dashboard", path: "/student/dashboard"}
];

const Questions = () => {
  const navigate = useNavigate();
  return (
    <div className="questions-layout">
      <div className="questions-sidebar">
        {sidebarButtons.map((btn) => (
          <button
            key={btn.label}
            className="questions-sidebar-btn"
            onClick={() => {
              navigate(btn.path);
            }}
          >
            {btn.label}
          </button>
        ))}
      </div>
      <div className="questions-main-area"></div>
      <div className="questions-chat-row">
        <button className="questions-chat-btn" onClick={() => navigate('/chat')}>Chat</button>
      </div>
    </div>
  );
};

export default Questions;
