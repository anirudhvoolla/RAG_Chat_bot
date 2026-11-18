const log = document.getElementById("log");
const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendButton = document.getElementById("send-button");
const welcome = document.getElementById("welcome");
let history = [];

function addMessage(cls, text, sources) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.textContent = text;
  if (sources) addSources(div, sources.split(", "));
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function addSources(message, sources) {
  if (!sources.length) return;
  const s = document.createElement("div");
  s.className = "sources";
  s.textContent = "sources: " + sources.join(", ");
  message.appendChild(s);
}

form.onsubmit = async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  if (welcome) welcome.remove();
  addMessage("user", text);
  const waiting = addMessage("bot", "Thinking...");
  waiting.classList.add("thinking");
  sendButton.disabled = true;
  try {
    const res = await fetch("/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, history }),
    });
    if (!res.ok) throw new Error(`Request failed (${res.status})`);
    if (!res.body) throw new Error("Streaming is not supported by this browser");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let answer = "";
    let sources = [];
    const handleEvent = (line) => {
      if (!line.trim()) return;
      const event = JSON.parse(line);
      if (event.type === "chunk") {
        answer += event.text;
        waiting.classList.remove("thinking");
        waiting.textContent = answer;
      } else if (event.type === "replace") {
        answer = event.text;
        waiting.classList.remove("thinking");
        waiting.textContent = answer;
      } else if (event.type === "sources") {
        sources = event.sources || [];
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      lines.forEach(handleEvent);
      if (done) break;
    }
    if (buffer.trim()) handleEvent(buffer);
    if (!answer) answer = "No answer was returned. Please try again.";
    waiting.classList.remove("thinking");
    waiting.textContent = answer;
    addSources(waiting, sources);
    history.push({ role: "user", content: text }, { role: "assistant", content: answer });
  } catch (err) {
    waiting.classList.remove("thinking");
    waiting.classList.add("error");
    waiting.textContent = "I couldn't get an answer just now. Please try again.";
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
};

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    form.requestSubmit();
  });
});
