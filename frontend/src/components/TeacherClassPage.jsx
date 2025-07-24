import React, { useState, useEffect } from "react";
import { useParams, useLocation } from "react-router-dom";

const mockStudents = [
  { id: 1, name: "Stud 1" },
  { id: 2, name: "Stud 2" },
  { id: 3, name: "Stud 3" },
  { id: 4, name: "Stud 4" },
  // Add or remove as needed
];

function chunkArray(array, perRow) {
  const result = [];
  for (let i = 0; i < array.length; i += perRow) {
    result.push(array.slice(i, i + perRow));
  }
  return result;
}

const sidebarBtns = [
  { label: "RAG" },
  { label: "Chat" },
  { label: "Questions" }
];

const TeacherClassPage = () => {
  const { classId } = useParams();
  const location = useLocation();

  // Use teacher details passed from previous page, fallback to empty object to avoid crashes
  const teacher = location.state?.teacher || {};

  const [students, setStudents] = useState([]);

  useEffect(() => {
    // Fetch students for this classId from backend
    // For now, just use mock data:
    setStudents(mockStudents);
  }, [classId]);

  const studentRows = chunkArray(students, 3);

  return (
    <div className="teacher-class-layout">
      {/* Sidebar */}
      <div className="teacher-class-sidebar">
        {sidebarBtns.map(btn => (
          <button className="teacher-class-sidebar-btn" key={btn.label}>
            {btn.label}
          </button>
        ))}
      </div>
      {/* Main area */}
      <div className="teacher-class-main">
        {/* Top bar */}
        <div className="teacher-class-topbar">
          <div style={{ flex: 1 }}></div>
          <div className="teacher-class-title">Unified Dashboard</div>
        </div>
        {/* Students grid */}
        <div className="teacher-class-students">
          {studentRows.map((row, idx) => (
            <div className="teacher-student-row" key={idx}>
              {row.map(s => (
                <div className="teacher-student-card" key={s.id}>
                  <div className="student-profile-icon">{s.name[0]}</div>
                  <div className="student-profile-name">{s.name}</div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TeacherClassPage;
