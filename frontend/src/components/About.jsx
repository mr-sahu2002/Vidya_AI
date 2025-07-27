import React from "react";
import "./About.css";
import AboutImg from "./About.png"; // Adjust path if needed

export default function About() {
  return (
    <div className="aboutContainer">
      <nav className="aboutNavbar">
        <div className="aboutLogo">VidyaAI</div>
        <div className="aboutNavLinks">
          <a href="#">Home</a>
          <a href="#">About</a>
          <a href="#">Pricing</a>
          <a href="#">Contact</a>
          <button className="getStartedBtn">Get Started</button>
          <button className="loginBtn">Log In</button>
        </div>
      </nav>
      <div className="aboutContent">
        <div className="aboutSection">
          <h2>About VidyaAI</h2>
          <p>
            VidyaAI is an innovative educational platform dedicated to transforming the learning experience for students and educators alike. Our mission is to provide accessible, high-quality educational resources that empower learners to achieve their full potential. We believe in the power of technology to enhance education and create a more engaging and effective learning environment.
          </p>
        </div>
        <div className="aboutImageWrapper">
          <img src={AboutImg} alt="About Illustration" className="aboutImage" />
        </div>
      </div>
      <section className="valuesSection">
        <div className="valuesRow">
          <div className="valueCard">
            <div className="valueTitle">Excellence</div>
            <div className="valueDesc">
              Commitment to high standards in every aspect of teaching and learning.
            </div>
          </div>
          <div className="valueCard">
            <div className="valueTitle">Community</div>
            <div className="valueDesc">
              Building a supportive and inclusive platform for diverse learners and educators.
            </div>
          </div>
          <div className="valueCard">
            <div className="valueTitle">Innovation</div>
            <div className="valueDesc">
              Leveraging the latest technology to promote creative and modern teaching.
            </div>
          </div>
          <div className="valueCard">
            <div className="valueTitle">Passion</div>
            <div className="valueDesc">
              Driven by a love for learning, teaching, and making an impact.
            </div>
          </div>
        </div>
        <div className="valuesFooter">
          Join us in our journey to revolutionize education and empower learners worldwide. Together, we can build a brighter future through knowledge and innovation.
        </div>
      </section>
    </div>
  );
}