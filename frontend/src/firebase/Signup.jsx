import React from 'react';
import { useNavigate } from "react-router-dom";
import { useState } from "react";
import { auth, db } from './firebase-config';
import { collection, addDoc, doc, setDoc } from 'firebase/firestore';

import {
  createUserWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
} from "firebase/auth";

const Signup = () => {
  const navigate = useNavigate();
  const [role, setRole] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  // Google authentication setup
  const googleProvider = new GoogleAuthProvider();

  // Function to create admin_user document in Firestore
  const createAdminUser = async (user, selectedRole) => {
    try {
      const adminUserData = {
        uid: user.uid,
        email: user.email,
        emailVerified: user.emailVerified,
        displayName: user.displayName || name, // Use displayName from Google or form name
        role: selectedRole,
        isAnonymous: user.isAnonymous,
        photoURL: user.photoURL || null,
        createdAt: new Date().toISOString(),
        lastLoginAt: new Date().toISOString(),
      };

      // Create document with user's UID as document ID
      await setDoc(doc(db, 'admin_users', user.uid), adminUserData);
      
      console.log('Admin user document created successfully');
      return adminUserData;
    } catch (error) {
      console.error('Error creating admin user document:', error);
      throw error;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!role || !name || !email || !username || !password) {
      alert("Please fill all fields");
      return;
    }

    try {
      // Create user with email and password
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      // Create admin_user document in Firestore
      await createAdminUser(user, role);

      if (role === "student") {
        navigate("/persona", { state: { name, username, email } });
      } else {
        navigate("/login");
      }
    } catch (error) {
      console.error('Error during signup:', error);
      alert(`Signup failed: ${error.message}`);
    }
  };

  const handleGoogleSignIn = async () => {
    try {
      // Check if user selected a role first
      if (!role) {
        alert("Please select your role (Student or Teacher) before signing in with Google");
        return;
      }

      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;
      
      console.log("User signed in with Google:", user);

      // Create admin_user document in Firestore
      await createAdminUser(user, role);
      
      // Navigate based on role
      if (role === "student") {
        navigate("/persona", { 
          state: { 
            name: user.displayName || name, 
            username: username || user.displayName, 
            email: user.email 
          } 
        });
      } else {
        navigate("/login");
      }
    } catch (error) {
      console.error('Error during Google sign-in:', error);
      alert(`Google sign-in failed: ${error.message}`);
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
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Enter your full name" required />
          </label>

          <label>
            Email ID
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Enter your email" required />
          </label>

          <label>
            Username
            <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Choose your username" required />
          </label>

          <label>
            Password
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Create a password" required />
          </label>

          <button type="submit" className="signup-btn">Create Account</button>

          {/* Google Sign-In Button */}
          <button type="button" onClick={handleGoogleSignIn} className="google-signin-button">
            Sign in with Google
          </button>
        </form>
      </div>
    </div>
  );
};

export default Signup;