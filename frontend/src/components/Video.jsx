import React from "react";
import { useNavigate } from "react-router-dom";
const sidebarButtons = [
  { label: "Notes", path: "/student/notes" },
  { label: "Games", path: "/student/games" },
  { label: "Assignment", path: "/student/assignment" },
  { label: "Questions", path: "/student/questions" },
  { label: "Dashboard", path: "/student/dashboard"}
];

const Video = () => {
  const navigate = useNavigate();
  return (
    <div className="video-layout">
      {/* Sidebar */}
      <div className="video-sidebar">
        {sidebarButtons.map((btn) => (
          <button
            key={btn.label}
            className="video-sidebar-btn"
            onClick={() => {
              navigate(btn.path);
            }}
          >
            {btn.label}
          </button>
        ))}
      </div>
      {/* Main area can be empty or hold your video content */}
      <div className="video-main-area"></div>

      {/* Chat button at the bottom center */}
      <div className="video-chat-row">
        <button className="questions-chat-btn" onClick={() => navigate('/chat')}>Chat</button>
      </div>
    </div>
  );
};

export default Video;
