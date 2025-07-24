import React from "react";
import { useNavigate } from "react-router-dom";

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="home-bg">
      <div className="home-content">
        <h1>
          Vidya <span className="highlight-blue">AI</span>
        </h1>
        <p className="subtitle">Empowering Teachers. Enriching Classrooms.</p>
        <div className="home-btn-group">
          <button className="btn-primary" onClick={() => navigate("/signup")}>
            Sign Up
          </button>
          <button className="btn-secondary" onClick={() => navigate("/login")}>
            Login
          </button>
        </div>
      </div>
    </div>
  );
};

export default Home;
