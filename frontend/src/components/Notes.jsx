import React from "react";
import { useNavigate } from "react-router-dom";
const sidebarButtons = [
  { label: "Video", path: "/student/video" },
  { label: "Games", path: "/student/games" },
  { label: "Assignment", path: "/student/assignment" },
  { label: "Questions", path: "/student/questions" },
  { label: "Dashboard", path: "/student/dashboard"}
];

const Notes = () => {
    const navigate = useNavigate();
  return (
    <div className="notes-layout">
      <div className="notes-sidebar">
        {sidebarButtons.map((btn) => (
          <button
            key={btn.label}
            className="notes-sidebar-btn"
            onClick={() => {
                navigate(btn.path);
            }}
          >
            {btn.label}
          </button>
        ))}
      </div>
      <div className="notes-main-area"></div>
      <div className="notes-chat-row">
        <button className="questions-chat-btn" onClick={() => navigate('/chat')}>Chat</button>
      </div>
    </div>
  );
};

export default Notes;
