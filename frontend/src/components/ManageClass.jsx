import React, { useRef, useState } from "react";
import { Bar } from "react-chartjs-2";
import "chart.js/auto";
import "./ManageClass.css";

// Mock user info
const user = {
  name: "Asha Sharma",
  email: "asha.sharma@school.edu",
};

// Sample hardcoded student performance data
const chartData = {
  labels: ["Alice", "Bob", "Charlie", "Daisy", "Elena", "Farhan"],
  datasets: [
    {
      label: "Average Score (%)",
      data: [85, 74, 92, 65, 88, 79],
      backgroundColor: "#6381f8",
      borderRadius: 7,
    },
  ],
};

const chartOptions = {
  plugins: {
    legend: { display: false },
    title: {
      display: true,
      text: "Consolidated Student Performance Report",
      font: { size: 20 },
    },
  },
  scales: {
    y: {
      beginAtZero: true,
      max: 100,
      title: { display: true, text: "Score (%)" },
      grid: { color: "#eceffd" },
    },
    x: {
      grid: { color: "#f5f6fa" },
    },
  },
};

export default function ManageClass() {
  const [showProfile, setShowProfile] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleBrowseClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="manageClassDashboard">
      <aside className="manageClassSidebar">
        <div className="sidebarLogo">
          <span className="emoji">📚</span>
          <span className="brand">VidyaAI</span>
        </div>
        <nav className="sidebarNav">
          <button className="sidebarBtn active">Manage Class</button>
        </nav>
      </aside>

      <main className="manageClassMain">
        <div className="manageClassHeader">
          <button
            className="profileAvatar"
            title="Profile"
            onClick={() => setShowProfile((v) => !v)}
          >
            {user.email.charAt(0).toUpperCase()}
          </button>
          {showProfile && (
            <div
              className="profilePopover"
              onClick={() => setShowProfile(false)}
            >
              <div className="profileInfoName">{user.name}</div>
              <div className="profileInfoEmail">{user.email}</div>
            </div>
          )}
        </div>

        <section className="manageClassContent">
          <div className="manageTabs">
            <button className="manageTab active">Upload Documents</button>
            <button className="manageTab">Performance Dashboard</button>
          </div>
          <div className="manageTabContent">
            {/* Upload Documents */}
            <div className="uploadDocBox">
              <h3>Upload Documents</h3>
              <p className="uploadDocDesc">
                Drag and drop or{" "}
                <span
                  className="browseBtn"
                  onClick={handleBrowseClick}
                  role="button"
                  tabIndex={0}
                >
                  browse
                </span>{" "}
                to upload. Supported: PDF.
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf"
                style={{ display: "none" }}
                onChange={handleFileChange}
              />
              {selectedFile && (
                <div className="uploadedFile">
                  Selected: <strong>{selectedFile.name}</strong>
                </div>
              )}
            </div>
            {/* Performance Dashboard (Chart) */}
            <div className="performanceDashboardBox">
              <Bar data={chartData} options={chartOptions} />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
