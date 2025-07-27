import React, { useState, useRef, useEffect } from "react";
import "./chat.css";

const DUMMY_PROFILE = {
  name: "John Doe",
  avatar: "https://ui-avatars.com/api/?name=John+Doe&background=5862fa&color=fff",
  role: "Student"
};

const NAV_LINKS = [
    { key: "whiteboard", label: "Whiteboard" },
  { key: "assignments", label: "Assignments" },
  { key: "questionnaire", label: "Questionnaire" },
  { key: "games", label: "Games" },
  { key: "chat", label: "Chat" },
  { key: "notes", label: "Notes" },
];

const Chat = () => {
  const [messages, setMessages] = useState([
    { sender: "llm", text: "Hello! How can I help you today?" }
  ]);
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    setMessages([...messages, { sender: "me", text: input }]);

    // Dummy LLM response, replace with real API logic
    setTimeout(() => {
      setMessages(curr => [
        ...curr,
        { sender: "llm", text: "You said: " + input }
      ]);
    }, 800);

    setInput("");
  };

  return (
    <div className="chat-layout">
      {/* Left Navigation */}
      <aside className="sidebar">
        <div className="profile">
          <img src={DUMMY_PROFILE.avatar} alt="Avatar" className="avatar" />
          <div className="profile-details">
            <div className="profile-name">{DUMMY_PROFILE.name}</div>
            <div className="profile-role">{DUMMY_PROFILE.role}</div>
          </div>
        </div>
        <nav className="nav-links">
          {NAV_LINKS.map((link, idx) => (
            <div className={`nav-link${link.label === "Chat" ? " active" : ""}`} key={idx}>
              <span className="nav-icon">{link.icon}</span>
              <span>{link.label}</span>
            </div>
          ))}
        </nav>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        <div className="chat-header">
          <span className="chat-icon">💬</span>
          <span className="chat-title">VidyaAi Chat</span>
        </div>
        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <div className={`chat-message ${msg.sender === "me" ? "outgoing" : "incoming"}`} key={idx}>
              <div className="msg-content">{msg.text}</div>
            </div>
          ))}
          <div ref={bottomRef}></div>
        </div>
        <form className="chat-input-bar" onSubmit={handleSend}>
          <input
            type="text"
            placeholder="Type your message…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            autoFocus
          />
          <button type="submit">Send</button>
        </form>
      </main>
    </div>
  );
};

export default Chat;
