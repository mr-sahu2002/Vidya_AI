import React from "react";
import "./notes.css";

const DUMMY_PROFILE = {
  name: "John Doe",
  avatar: "https://ui-avatars.com/api/?name=John+Doe&background=5862fa&color=fff",
  role: "Student"
};

const NAV_LINKS = [
  { key: "whiteboard", label: "Whiteboard" },
  { key: "assignments", label: "Assignments" },
  { key: "questionnaire", label: "Questionnaire" },
  { key: "games", label: "Games" },
  { key: "chat", label: "Chat" },
  { key: "notes", label: "Notes" },
];

const Notes = () => {
  return (
    <div className="notes-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="profile">
          <img src={DUMMY_PROFILE.avatar} alt="Avatar" className="avatar" />
          <div className="profile-details">
            <div className="profile-name">{DUMMY_PROFILE.name}</div>
            <div className="profile-role">{DUMMY_PROFILE.role}</div>
          </div>
        </div>
        <nav className="nav-links">
          {NAV_LINKS.map((link, idx) => (
            <div className={`nav-link${link.label === "Notes" ? " active" : ""}`} key={idx}>
              <span className="nav-icon">{link.icon}</span>
              <span>{link.label}</span>
            </div>
          ))}
        </nav>
      </aside>

      {/* Main Notes Section */}
      <main className="notes-main">
        <div className="notes-header">
          <span className="notes-icon">📑</span>
          <span className="notes-title">Notes</span>
        </div>
        <div className="notes-content">
          {/* 
            // TODO: Render PDF list and previews here.
            // This area will show uploaded PDFs from the backend.
          */}
          <div className="notes-placeholder">
            {/* Placeholder until backend is ready */}
            <p>PDF notes will appear here (to be loaded from the backend).</p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Notes;
