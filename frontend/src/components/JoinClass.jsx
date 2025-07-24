import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";

const JoinClass = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const username = location.state?.username || "";
  const [classCode, setClassCode] = useState("");

  const handleJoin = (e) => {
    e.preventDefault();

    if (!classCode.trim()) {
      alert("Please enter a class code.");
      return;
    }

    // In the future: validate classCode with backend here.
    navigate("/student/dashboard", { state: { username } });
  };

  return (
    <div className="joinclass-bg">
      <div className="joinclass-card">
        <h2 className="joinclass-title">join the class</h2>
        <form onSubmit={handleJoin} className="joinclass-form">
          <input
            type="text"
            placeholder="enter the code"
            className="joinclass-input"
            value={classCode}
            onChange={(e) => setClassCode(e.target.value)}
            required
          />
          <button type="submit" className="joinclass-btn">
            join
          </button>
        </form>
      </div>
    </div>
  );
};

export default JoinClass;
