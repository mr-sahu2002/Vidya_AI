import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";

const StudentDashboard = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const username = location.state?.username || "";
  const name = location.state?.name || "";

  const [showProfileBox, setShowProfileBox] = useState(false);

  const profile = {
    username,
    name,
    email: "",
    class: "",
    roll: ""
  };

  const handleRedirect = (path) => {
    navigate(path, { state: { username, name } });
  };

  const handleProfileClick = () => {
    setShowProfileBox((prev) => !prev);
  };

  return (
    <div className="student-dashboard-bg">
      <div className="student-dashboard-card">
        <div className="dashboard-header">
          <div className="dashboard-welcome">
            Welcome{" "}
            <span className="dashboard-student-name">{profile.name || profile.username}</span>!
          </div>

          <div className="profile-icon-area">
            <button className="profile-icon-btn" onClick={handleProfileClick}>
              <span className="material-icons" style={{ fontSize: "2.2rem" }}>
                account_circle
              </span>
            </button>
            {showProfileBox && (
              <div className="profile-popup">
                <strong>{profile.name || profile.username}</strong>
                <hr />
                <div>Email: {profile.email || "-"}</div>
                <div>Class: {profile.class || "-"}</div>
                <div>Roll No: {profile.roll || "-"}</div>
              </div>
            )}
          </div>
        </div>

        <div className="dashboard-buttons">
          <div className="dashboard-btn-row">
            <button onClick={() => navigate("/student/video")} className="dashboard-btn">Video</button>
            <button onClick={() => navigate("/student/notes")} className="dashboard-btn">Notes</button>
            <button onClick={() => navigate("/student/games")} className="dashboard-btn">Games</button>
          </div>
          <div className="dashboard-btn-row">
            <button onClick={() => navigate("/student/assignment")} className="dashboard-btn">Assignments</button>
            <button onClick={() => navigate("/student/questions")} className="dashboard-btn">Questions</button>
          </div>
          <div className="dashboard-btn-row">
            <button className="questions-chat-btn" onClick={() => navigate('/chat')}>Chat</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StudentDashboard;
