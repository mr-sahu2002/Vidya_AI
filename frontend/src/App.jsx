import React from "react";
import { Routes, Route } from "react-router-dom";

import Home from "./components/Home";
import Signup from "./components/Signup";
import Persona from "./components/Persona";
import LoginSelection from "./components/LoginSelection";
import CredentialForm from "./components/CredentialForm";
import JoinClass from "./components/JoinClass";
import StudentDashboard from "./components/StudentDashboard";
import Video from "./components/Video";
import Games from "./components/Games";
import Notes from "./components/Notes";
import Assignment from "./components/Assignment";
import Questions from "./components/Questions";
import TeacherDashboard from "./components/TeacherDashboard";
import TeacherClassPage from "./components/TeacherClassPage";
import Chat from "./components/Chat";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/persona" element={<Persona />} />
      <Route path="/login" element={<LoginSelection />} />
      <Route path="/login/:role" element={<CredentialForm />} />
      <Route path="/student/join-class" element={<JoinClass />} />
      <Route path="/student/dashboard" element={<StudentDashboard />} />
      <Route path="/student/video" element={<Video />} />
      <Route path="/student/games" element={<Games />} />
      <Route path="/student/notes" element={<Notes />} />
      <Route path="/student/assignment" element={<Assignment />} />
      <Route path="/student/questions" element={<Questions />} />
      <Route path="/teacher/dashboard" element={<TeacherDashboard />} />
      <Route path="/teacher/class/:classId" element={<TeacherClassPage />} />
      <Route path="/chat" element={<Chat />} />
    </Routes>
  );
}

export default App;
