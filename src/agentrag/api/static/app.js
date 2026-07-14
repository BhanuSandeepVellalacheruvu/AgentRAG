document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const queryInput = document.getElementById("query-input");
    const chatContainer = document.getElementById("chat-container");
    const sendButton = document.getElementById("send-button");

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // Append User Message
        appendMessage(query, "user");
        queryInput.value = "";
        queryInput.focus();

        // Show Typing Indicator
        const typingIndicator = appendTypingIndicator();
        chatContainer.scrollTop = chatContainer.scrollHeight;

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ query })
            });

            // Remove Typing Indicator
            typingIndicator.remove();

            if (!response.ok) {
                const errorData = await response.json();
                appendMessage(errorData.detail || "Server Error", "system", null, false);
                return;
            }

            const data = await response.json();
            appendMessage(data.reply, "system", data.steps, data.grounded);
        } catch (error) {
            typingIndicator.remove();
            appendMessage("Failed to connect to the server.", "system", null, false);
        }

        chatContainer.scrollTop = chatContainer.scrollHeight;
    });

    function appendMessage(text, sender, steps = null, grounded = false) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", `${sender}-message`);

        const avatarDiv = document.createElement("div");
        avatarDiv.classList.add("avatar");

        if (sender === "user") {
            avatarDiv.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`;
        } else {
            avatarDiv.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h4l2-9 4 18 2-9h4"/></svg>`;
        }

        const contentDiv = document.createElement("div");
        contentDiv.classList.add("content");

        const p = document.createElement("p");
        p.innerText = text;
        contentDiv.appendChild(p);

        if (steps && steps.length > 0) {
            const stepsDiv = document.createElement("div");
            stepsDiv.classList.add("steps-container");
            
            // Build audit trail list
            let trailHtml = `<strong>Agent Steps:</strong><br>`;
            steps.forEach(step => {
                trailHtml += `<span class="step-badge">${step}</span>`;
            });

            // Critique status
            if (grounded) {
                trailHtml += `<span class="step-badge grounded-badge">Grounded</span>`;
            } else {
                // Groundedness only matters if retrieval was used
                if (steps.includes("retrieved_context")) {
                    trailHtml += `<span class="step-badge ungrounded-badge">Hallucination Detected</span>`;
                }
            }

            stepsDiv.innerHTML = trailHtml;
            contentDiv.appendChild(stepsDiv);
        }

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function appendTypingIndicator() {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", "system-message");

        const avatarDiv = document.createElement("div");
        avatarDiv.classList.add("avatar");
        avatarDiv.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h4l2-9 4 18 2-9h4"/></svg>`;

        const contentDiv = document.createElement("div");
        contentDiv.classList.add("content");

        const typingDiv = document.createElement("div");
        typingDiv.classList.add("typing-indicator");
        typingDiv.innerHTML = `<span></span><span></span><span></span>`;

        contentDiv.appendChild(typingDiv);
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        chatContainer.appendChild(messageDiv);
        return messageDiv;
    }

    async function updateStatus() {
        const statusIndicator = document.getElementById("status-indicator");
        if (!statusIndicator) return;
        try {
            const response = await fetch("/api/status");
            if (response.ok) {
                const data = await response.json();
                if (data.use_mock_llm) {
                    statusIndicator.innerHTML = '<span class="pulse mock-pulse"></span> Offline (Mock Mode)';
                    statusIndicator.classList.add("mock-mode");
                } else {
                    statusIndicator.innerHTML = '<span class="pulse"></span> Online (AWS Bedrock)';
                    statusIndicator.classList.remove("mock-mode");
                }
            } else {
                statusIndicator.innerHTML = '<span class="pulse error-pulse"></span> Offline (Error)';
            }
        } catch (error) {
            statusIndicator.innerHTML = '<span class="pulse error-pulse"></span> Offline (Error)';
        }
    }
    updateStatus();
});
