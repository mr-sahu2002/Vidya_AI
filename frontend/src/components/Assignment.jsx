import React from "react";
import { useNavigate } from "react-router-dom";
const sidebarButtons = [
  { label: "Video", path: "/student/video" },
  { label: "Notes", path: "/student/notes" },
  { label: "Games", path: "/student/games" },
  { label: "Questions", path: "/student/questions" },
  { label: "Dashboard", path: "/student/dashboard"}
];

const Assignment = () => {
    const navigate = useNavigate();
  return (
    <div className="assignment-layout">
      <div className="assignment-sidebar">
        {sidebarButtons.map((btn) => (
          <button
            key={btn.label}
            className="assignment-sidebar-btn"
            onClick={() => {
              navigate(btn.path);
            }}
          >
            {btn.label}
          </button>
        ))}
      </div>
      <div className="assignment-main-area"></div>
      <div className="assignment-chat-row">
        <button className="questions-chat-btn" onClick={() => navigate('/chat')}>Chat</button>
      </div>
    </div>
  );
};

export default Assignment;
