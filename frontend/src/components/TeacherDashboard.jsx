import React, { useState } from "react";
import "./TeacherDashboard.css";

// Sidebar buttons (customize as needed)
const SIDEBAR_ITEMS = [
  { key: "dashboard", label: "Dashboard" },
  { key: "classes", label: "My Classes" },
  { key: "assignments", label: "Assignments" },
  { key: "chat", label: "Chat" },
  { key: "resources", label: "Resources" },
];

const user = {
  name: "Asha Sharma",
  email: "asha.sharma@school.edu",
};

function generateClassCode() {
  // simple 6-digit mock code
  return Math.floor(Math.random() * 900000 + 100000).toString();
}

export default function TeacherDashboard() {
  const [showProfile, setShowProfile] = useState(false);
  const [showCreateClass, setShowCreateClass] = useState(false);
  const [className, setClassName] = useState("");
  const [subjectName, setSubjectName] = useState("");
  const [error, setError] = useState("");
  const [classes, setClasses] = useState([]);

  const openProfile = () => setShowProfile((s) => !s);
  const closeProfile = () => setShowProfile(false);

  const handleOpenCreateClass = () => {
    setShowCreateClass(true);
    setClassName("");
    setSubjectName("");
    setError("");
  };

  const handleCreateClass = (e) => {
    e.preventDefault();
    if (!className.trim() || !subjectName.trim()) {
      setError("Please fill both fields.");
      return;
    }
    // Would send {className, subjectName} to backend here!
    const newClass = {
      className,
      subjectName,
      classCode: generateClassCode(), // Replace with real backend code in production
    };
    setClasses((c) => [newClass, ...c]);
    setShowCreateClass(false);
  };

  // Close modal on overlay click
  const handleOverlayClick = (e) => {
    if (e.target.className === "classModalOverlay") setShowCreateClass(false);
  };

  return (
    <div className="teacherDashboard">
      {/* Sidebar */}
      <div className="teacherSidebar">
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
      {/* Main area */}
      <div className="dashboardMain">
        <div className="dashboardHeader">
          <div className="headerRight">
            <button
              className="profileAvatar"
              onClick={openProfile}
              title="Profile"
            >
              {user.email.charAt(0).toUpperCase()}
            </button>
            {showProfile && (
              <div className="profilePopover" onClick={closeProfile}>
                <div className="profileInfoName">{user.name}</div>
                <div className="profileInfoEmail">{user.email}</div>
              </div>
            )}
          </div>
        </div>
        <div className="mainContent">
          <button className="createClassBtn" onClick={handleOpenCreateClass}>
            Create Class +
          </button>
          {/* Classes list */}
          <div className="classesList">
            {classes.length === 0 && (
              <div className="noClassMsg">
                No classes created yet. Click "Create Class" to get started.
              </div>
            )}
            {classes.map((cls, idx) => (
              <div className="classCard" key={idx}>
                <div className="classInfoBlock">
                  <div className="className">{cls.className}</div>
                  <div className="subjectName">{cls.subjectName}</div>
                </div>
                <div className="classCode">Class Code: <b>{cls.classCode}</b></div>
              </div>
            ))}
          </div>
        </div>
      </div>
      {/* Modal for class creation */}
      {showCreateClass && (
        <div className="classModalOverlay" onClick={handleOverlayClick}>
          <form className="classModal" onSubmit={handleCreateClass}>
            <h3>Create New Class</h3>
            <label>
              Class Name
              <input
                type="text"
                value={className}
                onChange={e => setClassName(e.target.value)}
                placeholder="Enter class name"
                autoFocus
              />
            </label>
            <label>
              Subject Name
              <input
                type="text"
                value={subjectName}
                onChange={e => setSubjectName(e.target.value)}
                placeholder="Enter subject name"
              />
            </label>
            {error && <div className="modalError">{error}</div>}
            <div className="modalBtns">
              <button
                type="button"
                className="modalCancel"
                onClick={() => setShowCreateClass(false)}
              >
                Cancel
              </button>
              <button type="submit" className="modalSubmit">
                Create
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}