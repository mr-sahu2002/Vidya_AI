// // firebase-config.js
// import { initializeApp } from "firebase/app";
// import { getAuth } from "firebase/auth";
// import { getAnalytics } from "firebase/analytics";

// const firebaseConfig = {
//   apiKey: "AIzaSyAL03U1d8vAMDg8vtcuWB29HmBVVPRhn2c",
//   authDomain: "vidyaai-1dcfa.firebaseapp.com",
//   projectId: "vidyaai-1dcfa",
//   storageBucket: "vidyaai-1dcfa.firebasestorage.app",
//   messagingSenderId: "1020506839904",
//   appId: "1:1020506839904:web:4cd95ceff2cdb304955c05",
//   measurementId: "G-D0SYWSNDYC"
// };

// // Initialize Firebase
// const app = initializeApp(firebaseConfig);
// const analytics = getAnalytics(app); // optional

// // Export auth object
// export const auth = getAuth(app);



// firebase-config.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyC-XqtIhL7DnS16NcH5rnOXLkGVBNwvLD0",
  authDomain: "vidya-ai-cdb33.firebaseapp.com",
  projectId: "vidya-ai-cdb33",
  storageBucket: "vidya-ai-cdb33.firebasestorage.app",
  messagingSenderId: "547472003451",
  appId: "1:547472003451:web:04465ce4cfa2212799adde",
  measurementId: "G-Y04BMHTSFQ"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app); // optional

// Export auth object
export const auth = getAuth(app);

