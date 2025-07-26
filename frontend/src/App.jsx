import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from "react-router-dom";
import { onAuthStateChanged } from 'firebase/auth';
import Home from "./components/Home";
import { auth } from './firebase/firebase-config';
import Signup from "./firebase/Signup";
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

const App = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      console.log('Auth state changed:', currentUser); // Debug log
      setUser(currentUser);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  // Show loading screen while checking authentication
  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh' 
      }}>
        Loading...
      </div>
    );
  }

  // If user is not authenticated, only show signup route
  if (!user) {
    return (
      <Routes>
        <Route path="/" element={<Signup />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    );
  }

  // If user is authenticated, show all routes
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/persona" replace />} />
      <Route path="/persona" element={<Persona />} />
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
      <Route path="*" element={<Navigate to="/persona" replace />} />
    </Routes>
  );
};

export default App;