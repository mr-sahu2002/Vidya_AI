// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import About from './components/About';
import Pricing from './components/Pricing';
import Login from './components/Login';
import TeacherDashboard from './components/TeacherDashboard';
import StudentDashboard from './components/StudentDashboard';
import ClassDashboard from './components/ClassDashboard';
import ManageClass from './components/ManageClass';
import Whiteboard from './components/Whiteboard';
import Chat from './components/chat';
import Notes from './components/notes';
import Questions from './components/Questions';
import StudentGames from './components/StudentGames';
import Persona from './components/Persona';
import { onAuthStateChanged } from 'firebase/auth';
import { auth } from './firebase/firebase-config';
import Signup from "./firebase/Signup";



function App() {
  return (
    
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/about" element={<About />} />
        <Route path="/pricing" element={<Pricing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        
        {/* Teacher Routes */}
        <Route path="/teacher/dashboard" element={<TeacherDashboard />} />
        <Route path="/teacher/class/:classId/manage" element={<ManageClass />} />
        
        {/* Student Routes */}
        <Route path="/student/classes" element={<ClassDashboard />} />
        <Route path="/student/dashboard/:classId" element={<StudentDashboard />} />
        <Route path="/student/class/:classId/whiteboard" element={<Whiteboard />} />
        <Route path="/student/class/:classId/chat" element={<Chat />} />
        <Route path="/student/class/:classId/notes" element={<Notes />} />
        <Route path="/student/class/:classId/questions" element={<Questions />} />
        <Route path="/student/class/:classId/games" element={<StudentGames />} />
        <Route path="/student/persona" element={<Persona />} />
        
        {/* Fallback Route */}
        <Route path="*" element={<LandingPage />} />
      </Routes>
    
  );
}

export default App;