import React, { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";

// Simulate fetching classes from backend
const mockFetchClasses = () =>
  new Promise((resolve) => {
    setTimeout(() => {
      resolve([
       
      ]);
    }, 500);
  });

// Helper to chunk array into rows of `perRow` length
function chunkArray(array, perRow) {
  const result = [];
  for (let i = 0; i < array.length; i += perRow) {
    result.push(array.slice(i, i + perRow));
  }
  return result;
}

const TeacherDashboard = () => {
  const location = useLocation();
  const navigate = useNavigate();

  // Get teacher details from navigation state
  const username = location.state?.username || "Teacher";
  const name = location.state?.name || "";
  const email = location.state?.email || "";

  const [classes, setClasses] = useState([]);
  const [showAddBox, setShowAddBox] = useState(false);
  const [newClass, setNewClass] = useState({ name: "", section: "", code: "" });
  const [error, setError] = useState("");

  useEffect(() => {
    mockFetchClasses().then(setClasses);
  }, []);

  const handleAddClass = (e) => {
    e.preventDefault();
    if (!newClass.name.trim() || !newClass.section.trim() || !newClass.code.trim()) {
      setError("All fields are required.");
      return;
    }
    setClasses(prev => [
      ...prev,
      {
        id: Date.now(),
        name: newClass.name,
        section: newClass.section,
        code: newClass.code
      }
    ]);
    setNewClass({ name: "", section: "", code: "" });
    setShowAddBox(false);
    setError("");
  };

  // Arrange classes into rows of 4 buttons + + button at the end
  const classRows = chunkArray(classes, 4);
  if (classRows.length === 0) classRows.push([]);
  if (classRows[classRows.length - 1].length === 4) classRows.push([]);
  classRows[classRows.length - 1].push("add");

  return (
    <div className="teacher-dashboard-bg">
      <div className="teacher-dashboard-container">
        <div className="teacher-dashboard-header">
          Welcome <b>{username}</b>!
        </div>

        <div className="teacher-class-list-matrix">
          {classRows.map((row, rowIdx) => (
            <div className="teacher-class-list-row" key={rowIdx}>
              {row.map((cls) =>
                cls === "add" ? (
                  <button
                    key="add"
                    className="teacher-class-btn add-btn"
                    onClick={() => setShowAddBox(true)}
                  >
                    +
                  </button>
                ) : (
                  <button
                    key={cls.id}
                    className="teacher-class-btn"
                    onClick={() =>
                      navigate(`/teacher/class/${cls.id}`, {
                        state: {
                          classInfo: cls,
                          teacherUsername: username,
                          teacherName: name,
                          teacherEmail: email
                        }
                      })
                    }
                  >
                    {cls.name}
                  </button>
                )
              )}
            </div>
          ))}
        </div>

        {showAddBox && (
          <div className="add-class-modal-bg">
            <div className="add-class-modal">
              <h3>Create new class</h3>
              <form onSubmit={handleAddClass} className="add-class-form">
                <input
                  type="text"
                  placeholder="Class Name"
                  value={newClass.name}
                  onChange={e => setNewClass(n => ({ ...n, name: e.target.value }))}
                />
                <input
                  type="text"
                  placeholder="Section"
                  value={newClass.section}
                  onChange={e => setNewClass(n => ({ ...n, section: e.target.value }))}
                />
                <input
                  type="text"
                  placeholder="Class Code"
                  value={newClass.code}
                  onChange={e => setNewClass(n => ({ ...n, code: e.target.value }))}
                />
                <div style={{ display: "flex", gap: "12px", marginTop: "10px" }}>
                  <button type="submit">Create</button>
                  <button
                    type="button"
                    className="cancel-btn"
                    onClick={() => {
                      setShowAddBox(false);
                      setError("");
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </form>
              {error && <div className="add-class-error">{error}</div>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TeacherDashboard;
