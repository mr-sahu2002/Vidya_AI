import React, { useState } from "react";
import "./StudentDashboard.css";

const SIDEBAR_ITEMS = [
  { key: "whiteboard", label: "Whiteboard" },
  { key: "assignments", label: "Assignments" },
  { key: "questionnaire", label: "Questionnaire" },
  { key: "games", label: "Games" },
  { key: "chat", label: "Chat" },
  { key: "notes", label: "Notes" },
];

const user = {
  name: "Sarah Lee",
  email: "sarah.lee@example.com",
  usn: "2CS22CS123",
};

export default function StudentDashboard() {
  const [showProfile, setShowProfile] = useState(false);

  const handleAvatarClick = () => setShowProfile((v) => !v);
  const handlePersonaClick = () => {
    // navigation: use react-router if you have it, else use window.location
    window.location.href = "/persona";
  };

  return (
    <div className="studentDashboard">
      {/* Sidebar */}
      <div className="studentSidebar">
        <div className="sidebarLogo">
          <span className="emoji">📚</span>
          <span className="brand">VidyaAI</span>
        </div>
        <div className="sidebarNav">
          {SIDEBAR_ITEMS.map((item) => (
            <button className="sidebarBtn" key={item.key}>
              {item.label}
            </button>
          ))}
        </div>
      </div>
      {/* Main */}
      <div className="dashboardMain">
        <div className="dashboardHeader">
          <div className="headerRight">
            {/* Avatar and edit persona */}
            <div className="avatarAndPersona">
              <button
                className="profileAvatar"
                onClick={handleAvatarClick}
                title="Profile"
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
              <button className="editPersonaBtn" onClick={handlePersonaClick}>
                Edit Persona
              </button>
            </div>
          </div>
        </div>
        {/* Content */}
        <div className="mainContent">
          <h2 className="welcomeBack">Welcome back, Sarah!</h2>
          <p className="dashboardIntro">
            Let’s start with a new whiteboard session. You can also explore other features like assignments,
            questionnaires, games, chat, and notes from the sidebar.
          </p>
        </div>
      </div>
    </div>
  );
}