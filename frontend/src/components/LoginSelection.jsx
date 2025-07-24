import React from "react";
import { useNavigate } from "react-router-dom";

const LoginSelection = () => {
  const navigate = useNavigate();

  const handleSelection = (role) => {
    navigate(`/login/${role}`);
  };

  return (
    <div className="login-bg">
      <div className="login-card">
        <h2>Login As</h2>
        <div className="login-btn-group">
          <button className="login-btn student" onClick={() => handleSelection("student")}>
            Student
          </button>
          <button className="login-btn teacher" onClick={() => handleSelection("teacher")}>
            Teacher
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginSelection;
