// credential firebase-config.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyAL03U1d8vAMDg8vtcuWB29HmBVVPRhn2c",
  authDomain: "vidyaai-1dcfa.firebaseapp.com",
  projectId: "vidyaai-1dcfa",
  storageBucket: "vidyaai-1dcfa.firebasestorage.app",
  messagingSenderId: "1020506839904",
  appId: "1:1020506839904:web:4cd95ceff2cdb304955c05",
  measurementId: "G-D0SYWSNDYC"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase services
const db = getFirestore(app);
const auth = getAuth(app);
const analytics = getAnalytics(app); // Optional

// Export the instances
export {auth , db};


