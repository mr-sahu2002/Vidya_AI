import React from "react";
import { useNavigate } from "react-router-dom";
const sidebarButtons = [
  { label: "Video", path: "/student/video" },
  { label: "Notes", path: "/student/notes" },
  { label: "Assignment", path: "/student/assignment" },
  { label: "Questions", path: "/student/questions" },
  { label: "Dashboard", path: "/student/dashboard"}
];

const Games = () => {
    const navigate = useNavigate();
  return (
    <div className="games-layout">
      <div className="games-sidebar">
        {sidebarButtons.map((btn) => (
          <button
            key={btn.label}
            className="games-sidebar-btn"
            onClick={() => {
              navigate(btn.path);
            }}
          >
            {btn.label}
          </button>
        ))}
      </div>
      <div className="games-main-area"></div>
      <div className="games-chat-row">
        <button className="questions-chat-btn" onClick={() => navigate('/chat')}>Chat</button>
      </div>
    </div>
  );
};

export default Games;
