import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const Signup = () => {
  const navigate = useNavigate();
  const [role, setRole] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!role || !name || !email || !username || !password) {
      alert("Please fill all fields");
      return;
    }

    if (role === "student") {
      navigate("/login");
      
    } else {
      navigate("/login");
    }
  };

  return (
    <div className="signup-bg">
      <div className="signup-card">
        <h2>Create Account</h2>
        <form onSubmit={handleSubmit} className="signup-form">
          <label>
            I am a
            <select value={role} onChange={(e) => setRole(e.target.value)} required>
              <option value="">-- Select --</option>
              <option value="student">Student</option>
              <option value="teacher">Teacher</option>
            </select>
          </label>

          <label>
            Name
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter your full name"
              required
            />
          </label>

          <label>
            Email ID
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              required
            />
          </label>

          <label>
            Username
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Choose your username"
              required
            />
          </label>

          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Create a password"
              required
            />
          </label>

          <button type="submit" className="signup-btn">Create Account</button>
        </form>
      </div>
    </div>
  );
};

export default Signup;
