import React, { useState } from "react";
import "./ClassDashboard.css";

// Mock backend lookup for class codes and their info
// Replace this with your real backend API integration
const mockClasses = {
  "ABC123": { className: "Mathematics 101", subject: "Mathematics" },
  "ENG456": { className: "English Literature", subject: "English" },
  "SCI789": { className: "Physics Basics", subject: "Science" },
};

const user = {
  name: "John Doe",
  email: "john.doe@example.com",
};

export default function ClassDashboard() {
  const [showAddClass, setShowAddClass] = useState(false);
  const [classCodeInput, setClassCodeInput] = useState("");
  const [error, setError] = useState("");
  const [classes, setClasses] = useState([]);

  const handleProfileClick = () => {
    // Could add profile dropdown logic here if needed
  };

  const handleAddClassClick = () => {
    setShowAddClass((v) => !v);
    setClassCodeInput("");
    setError("");
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");

    const code = classCodeInput.trim().toUpperCase();

    if (!code) {
      setError("Please enter the class code.");
      return;
    }

    if (!mockClasses[code]) {
      setError("Wrong code entered.");
      return;
    }

    // Check if this class is already added
    if (classes.some((cls) => cls.classCode === code)) {
      setError("Class already added.");
      return;
    }

    // Add the class info to the list
    const classInfo = { ...mockClasses[code], classCode: code };
    setClasses((prev) => [...prev, classInfo]);
    setClassCodeInput("");
    setShowAddClass(false);
  };

  return (
    <div className="classDashboard">
      <header className="dashboardHeader">
        <div className="profileContainer">
          <button
            className="profileAvatar"
            onClick={handleProfileClick}
            title="Profile"
            aria-label="User profile"
          >
            {user.email.charAt(0).toUpperCase()}
          </button>
          <div className="userInfo">
            <div className="userName">{user.name}</div>
            <div className="userEmail">{user.email}</div>
          </div>
        </div>
        <button className="addClassBtn" onClick={handleAddClassClick}>
          {showAddClass ? "Cancel" : "Add Class"}
        </button>
      </header>

      {showAddClass && (
        <form className="addClassForm" onSubmit={handleSubmit} noValidate>
          <label htmlFor="classCodeInput">Enter Class Code</label>
          <input
            id="classCodeInput"
            type="text"
            value={classCodeInput}
            onChange={(e) => setClassCodeInput(e.target.value)}
            placeholder="e.g. ABC123"
            autoFocus
          />
          <button type="submit">Submit</button>
          {error && <div className="errorMsg">{error}</div>}
        </form>
      )}

      <main className="classesContainer">
        {classes.length === 0 ? (
          <div className="noClassesMsg">No classes added.</div>
        ) : (
          classes.map(({ className, subject, classCode }) => (
            <button className="classButton" key={classCode}>
              <div className="classDetails">
                <div className="className">{className}</div>
                <div className="subjectName">{subject}</div>
              </div>
              <div className="classCode">Code: {classCode}</div>
            </button>
          ))
        )}
      </main>
    </div>
  );
}