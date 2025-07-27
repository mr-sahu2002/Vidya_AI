import React, { useState } from "react";
import "./StudentGames.css";

// List of sidebar features
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

export default function StudentGames() {
  const [showProfile, setShowProfile] = useState(false);

  return (
    <div className="gamesDashboard">
      {/* Sidebar */}
      <aside className="gamesSidebar">
        <div className="sidebarLogo">
          <span className="emoji">📚</span>
          <span className="brand">VidyaAI</span>
        </div>
        <nav className="sidebarNav">
          {SIDEBAR_ITEMS.map((item) => (
            <button
              key={item.key}
              className={`sidebarBtn${item.key === "games" ? " active" : ""}`}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      {/* Main Section */}
      <main className="gamesMain">
        {/* Top right profile */}
        <div className="gamesHeader">
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

        {/* Games Content */}
        <section className="gamesContent">
          <div className="gamesTitle">Games</div>
          <div className="gamesSub">Checking for today's lesson plan...</div>
          <div className="gamesLoadingBarWrap">
            <div className="gamesLoadingBar" />
          </div>
          <div className="gamesEmpty">
            <img
              src="https://images.unsplash.com/photo-1506744038136-46273834b3fb?fit=crop&w=400&q=80"
              alt="No lesson plan"
              className="gamesEmptyImg"
              draggable="false"
            />
            <div className="gamesEmptyMain">No lesson plan has been created for today.</div>
            <div className="gamesEmptySub">Please check back later!</div>
            <button className="gamesRefreshBtn">Refresh</button>
          </div>
        </section>
      </main>
    </div>
  );
}