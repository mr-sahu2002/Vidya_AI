import React from 'react';
import './LandingPage.css';

function LandingPage() {
  return (
    <div className="landing-root">
      <header className="landing-header">
        <div className="logo">
          <span role="img" aria-label="books" className="logo-icon">📚</span>
          VidyaAI
        </div>
        <nav className="nav-links">
          <a href="#pricing">Pricing</a>
          <a href="#about">About</a>
          <button className="signup-btn">Sign Up</button>
          <button className="login-btn">Login</button>
        </nav>
      </header>

      <main className="landing-main">
        <div className="main-bg">
          <img src="./land.png" alt="Background" className="bg-image"/>
          <div className="main-content">
            <h1>Empowering Education with AI</h1>
            <p>
              VidyaAI is an innovative platform designed to enhance the learning experience for students and educators alike. Our AI-driven tools provide personalized learning paths, insightful analytics, and collaborative features to foster a dynamic educational environment.
            </p>
            <button className="main-cta">Get Started</button>
          </div>
        </div>
      </main>

      <footer className="landing-footer">
        <a href="#terms">Terms of Service</a>
        <a href="#privacy">Privacy Policy</a>
        <a href="#contact">Contact Us</a>
        <div className="copyright">
          &#64;2024 VidyaAI. All rights reserved.
        </div>
      </footer>
    </div>
  );
}

export default LandingPage;
