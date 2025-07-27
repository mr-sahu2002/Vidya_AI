import React, { useState } from "react";
import "./Login.css";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [redirect, setRedirect] = useState(false);

  const handleLogin = (e) => {
    e.preventDefault();
    setError("");
    // Simulated validation
    const isValid = email === "user@demo.com" && password === "password123";
    if (isValid) {
      setRedirect(true);
    } else {
      setError("Email ID or password is wrong");
    }
  };

  if (redirect) {
    return (
      <div className="nextPage">
        <h1>Welcome, {email}!</h1>
        <p>You have logged in successfully.</p>
      </div>
    );
  }

  return (
    <div className="container">
      <nav className="navbar">
        <span className="logo">VidyaAI</span>
        <div className="navLinks">
          <a href="#">Home</a>
          <a href="#">About</a>
          <a href="#">Contact</a>
          <button className="signUpBtn">Sign Up</button>
        </div>
      </nav>

      <form className="form" onSubmit={handleLogin}>
        <h2 className="title">Welcome back</h2>
        <div className="inputGroup">
          <label>Email ID</label>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="inputGroup">
          <label>Password</label>
          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit" className="loginBtn">
          Login
        </button>
        {error && <div className="errorMsg">{error}</div>}
      </form>
    </div>
  );
}