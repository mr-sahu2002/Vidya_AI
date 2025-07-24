import React, { useState } from "react";

const Chat = () => {
  const [message, setMessage] = useState("");
  const [chatLog, setChatLog] = useState([]);

  const handleInputChange = (e) => {
    setMessage(e.target.value);
  };

  const handleSend = () => {
    if (!message.trim()) return;
    setChatLog([...chatLog, { text: message }]);
    setMessage("");
  };

  const handleInputKeyDown = (e) => {
    if (e.key === "Enter") handleSend();
  };

  return (
    <div className="chat-page-bg">
      <div className="chat-card">
        <div className="chat-header">
          Vidya AI
        </div>
        <div className="chat-log">
          {chatLog.length === 0 && (
            <div className="chat-placeholder"></div>
          )}
          {chatLog.map((msg, idx) => (
            <div className="chat-msg" key={idx}>
              {msg.text}
            </div>
          ))}
        </div>
        <div className="chat-input-row">
          <input
            type="text"
            className="chat-input"
            placeholder="Type your message here ..."
            value={message}
            onChange={handleInputChange}
            onKeyDown={handleInputKeyDown}
          />
          <button className="chat-send-btn" onClick={handleSend}>
            {/* Updated button content below via CSS */}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chat;
