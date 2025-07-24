import React, { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

const CredentialForm = () => {
  const { role } = useParams();
  const navigate = useNavigate();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  // Assuming you have logic to retrieve or hardcode teacher's name and email
  // This would typically come from your backend/auth system.
  const teacherName = "Jane Smith"; 
  const teacherEmail = "jane.smith@school.edu";

  const handleSubmit = (e) => {
    e.preventDefault();

    if (username.trim() !== "" && password.trim() !== "") {
      if (role === "student") {
        navigate("/student/join-class", { state: { username } });
      } else if (role === "teacher") {
        navigate("/teacher/dashboard", { state: { username, name: teacherName, email: teacherEmail } });
      }
    } else {
      alert("Please enter valid credentials.");
    }
  };

  return (
    <div className="login-bg">
      <div className="login-card">
        <h2>
          Login as{" "}
          <span className="role-highlight">
            {role.charAt(0).toUpperCase() + role.slice(1)}
          </span>
        </h2>
        <form className="login-form" onSubmit={handleSubmit}>
          <label htmlFor="username" className="input-label">
            Username
          </label>
          <input
            id="username"
            type="text"
            placeholder="Enter your username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />

          <label htmlFor="password" className="input-label">
            Password
          </label>
          <input
            id="password"
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <button type="submit" className="login-submit">
            Login
          </button>
        </form>
      </div>
    </div>
  );
};

export default CredentialForm;
