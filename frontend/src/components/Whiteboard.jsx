import React, { useState } from "react";
import "./Whiteboard.css";

// Use any sidebar items you like
const SIDEBAR_ITEMS = [
  { key: "whiteboard", label: "Whiteboard" },
  { key: "assignments", label: "Assignments" },
  { key: "questionnaire", label: "Questionnaire" },
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

// Example images (add your own image URLs or imports)
const images = [
  "https://images.unsplash.com/photo-1506744038136-46273834b3fb?fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1465101046530-73398c7f28ca?fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1505678261036-a3fcc5e884ee?fit=crop&w=400&q=80",
  "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?fit=crop&w=400&q=80",
];

export default function Whiteboard() {
  const [showProfile, setShowProfile] = useState(false);

  return (
    <div className="whiteboardDashboard">
      {/* Sidebar */}
      <aside className="whiteboardSidebar">
        <div className="sidebarLogo">
          <span className="emoji">📚</span>
          <span className="brand">VidyaAI</span>
        </div>
        <nav className="sidebarNav">
          {SIDEBAR_ITEMS.map((item) => (
            <button
              key={item.key}
             className={`sidebarBtn${item.key === "whiteboard" ? " active" : ""}`}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      {/* Main Section */}
      <main className="whiteboardMain">
        {/* Top right profile */}
        <div className="whiteboardHeader">
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

        {/* Whiteboard Content */}
        <section className="whiteboardContent">
          <div className="whiteboardTitle">Whiteboard</div>
          <div className="whiteboardInput" />
          <div className="whiteboardImgScrollWrap">
            <div className="whiteboardImgScroll">
              {images.map((src, idx) => (
                <div className="whiteboardImgItem" key={idx}>
                 <img src={src} alt={`Whiteboard ${idx + 1}`} />
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}