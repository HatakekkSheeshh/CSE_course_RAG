import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

type SourceChunk = {
  chunk_id: string;
  text: string;
  score: number;
  confidence: number;
  course?: string | null;
  metadata?: Record<string, unknown> | null;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: SourceChunk[];
  timestamp: number;
};

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";

// Derive API root (backend base URL without /api)
const API_ROOT = API_BASE_URL.replace(/\/api\/?$/, "");

function formatCourseName(courseId: string): string {
  // Simple formatting: replace underscores with spaces
  return courseId.replace(/_/g, " ");
}

// Simple markdown parser for bold text
function parseMarkdown(text: string): ReactNode {
  // Split by **text** pattern
  const parts: ReactNode[] = [];
  const regex = /\*\*(.*?)\*\*/g;
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = regex.exec(text)) !== null) {
    // Add text before match
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }
    // Add bold text
    parts.push(<strong key={key++}>{match[1]}</strong>);
    lastIndex = regex.lastIndex;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  return parts.length > 0 ? parts : text;
}

// Generate session ID (persist across page reloads)
function getSessionId(): string {
  const stored = localStorage.getItem("rag_session_id");
  if (stored) {
    return stored;
  }
  const newId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  localStorage.setItem("rag_session_id", newId);
  return newId;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [course, setCourse] = useState<string | null>(null);
  const [courses, setCourses] = useState<string[]>([]);
  const [useStreaming, setUseStreaming] = useState(true);
  const [sessionId, setSessionId] = useState(() => getSessionId());
  const [startNewConversation, setStartNewConversation] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const canSubmit = useMemo(
    () => question.trim().length > 0 && !loading,
    [question, loading]
  );

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const askQuestionStreaming = useCallback(async () => {
    if (!canSubmit) {
      return;
    }

    const userQuestion = question.trim();
    setLoading(true);
    setError(null);
    setQuestion("");

    // Add user message
    const userMessage: Message = {
      role: "user",
      content: userQuestion,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMessage]);

    // Add placeholder assistant message for streaming
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: "",
        timestamp: Date.now(),
      },
    ]);

    try {
      const response = await fetch(`${API_BASE_URL}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userQuestion,
          course: course,
          top_k: 5,
          session_id: sessionId,
          start_new_conversation: startNewConversation,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to connect to server");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let sources: SourceChunk[] = [];
      let fullAnswer = "";

      if (!reader) {
        throw new Error("No response body");
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          // Skip empty lines
          if (!line.trim()) continue;
          
          if (line.startsWith("data: ")) {
            try {
              const jsonStr = line.slice(6).trim();
              if (!jsonStr) continue; // Skip empty data
              const data = JSON.parse(jsonStr);

              if (data.type === "sources") {
                sources = data.sources || [];
                // Update message with sources (create new object to trigger re-render)
                setMessages((prev) => {
                  const updated = [...prev];
                  const lastIndex = updated.length - 1;
                  if (lastIndex >= 0 && updated[lastIndex].role === "assistant") {
                    updated[lastIndex] = {
                      ...updated[lastIndex],
                      sources: sources,
                    };
                  }
                  return updated;
                });
              } else if (data.type === "token") {
                fullAnswer += data.token;
                // Debug: log token received
                console.log("Token received:", data.token, "Full answer so far:", fullAnswer);
                // Update last message with streaming content (create new object)
                setMessages((prev) => {
                  const updated = [...prev];
                  const lastIndex = updated.length - 1;
                  if (lastIndex >= 0 && updated[lastIndex].role === "assistant") {
                    updated[lastIndex] = {
                      ...updated[lastIndex],
                      content: fullAnswer,
                    };
                  }
                  return updated;
                });
              } else if (data.type === "done") {
                // Final answer
                if (data.answer && data.answer !== fullAnswer) {
                  fullAnswer = data.answer;
                }
                setMessages((prev) => {
                  const updated = [...prev];
                  const lastIndex = updated.length - 1;
                  if (lastIndex >= 0 && updated[lastIndex].role === "assistant") {
                    updated[lastIndex] = {
                      ...updated[lastIndex],
                      content: fullAnswer,
                      sources: sources,
                    };
                  }
                  return updated;
                });
              } else if (data.type === "error") {
                throw new Error(data.message || "Unknown error");
              }
            } catch (e) {
              console.error("Error parsing SSE data:", e);
            }
          }
        }
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to reach backend.";
      setError(message);
      // Remove the placeholder assistant message on error
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
    // Reset flag after first request
    if (startNewConversation) {
      setStartNewConversation(false);
    }
  }, [canSubmit, question, course, sessionId, messages.length, startNewConversation]);

  const askQuestionNonStreaming = useCallback(async () => {
    if (!canSubmit) {
      return;
    }

    const userQuestion = question.trim();
    setLoading(true);
    setError(null);
    setQuestion("");

    // Add user message
    const userMessage: Message = {
      role: "user",
      content: userQuestion,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userQuestion,
          course: course,
          top_k: 5,
          session_id: sessionId,
          start_new_conversation: startNewConversation,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Unexpected server error");
      }

      if (data.status !== "ok") {
        setError(data.answer ?? "No answer available.");
        return;
      }

      // Add assistant message
      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        sources: data.sources ?? [],
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to reach backend.";
      setError(message);
    } finally {
      setLoading(false);
    }
    // Reset flag after first request
    if (startNewConversation) {
      setStartNewConversation(false);
    }
  }, [canSubmit, question, course, sessionId, startNewConversation]);

  const askQuestion = useCallback(() => {
    if (useStreaming) {
      void askQuestionStreaming();
    } else {
      void askQuestionNonStreaming();
    }
  }, [useStreaming, askQuestionStreaming, askQuestionNonStreaming]);

  const clearHistory = useCallback(() => {
    setMessages([]);
    // Clear session ID to start fresh conversation
    localStorage.removeItem("rag_session_id");
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem("rag_session_id", newSessionId);
    setSessionId(newSessionId);
    // Set flag to tell backend to start new conversation
    setStartNewConversation(true);
  }, []);

  // load course
  useEffect(() => {
    const loadCourses = async () => {
      try {
        const response = await fetch(`${API_ROOT}/courses`);
        if (!response.ok) {
          return;
        }
        const data = await response.json();
        if (Array.isArray(data.courses)) {
          setCourses(data.courses);
        }
      } catch {
      }
    };

    void loadCourses();
  }, []);

  return (
    <div className="min-h-screen bg-[#131316] text-white flex flex-col">
      <div className="flex items-center justify-between px-6 py-4 text-sm border-b border-white/10">
        <div className="font-semibold tracking-wide">CSE Course RAG</div>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={useStreaming}
              onChange={(e) => setUseStreaming(e.target.checked)}
              className="w-4 h-4 rounded border-white/20 bg-[#18181f] text-indigo-600 focus:ring-indigo-500"
            />
            <span>Streaming</span>
          </label>
          {messages.length > 0 && (
            <button
              onClick={clearHistory}
              className="text-xs text-gray-400 hover:text-gray-300 px-2 py-1 rounded hover:bg-white/5 transition"
            >
              Clear History
            </button>
          )}
          <div className="text-xs text-gray-400">
            Powered by Retrieval-Augmented Generation
          </div>
        </div>
      </div>

      <main className="flex-1 flex flex-col items-center px-4 py-10 gap-6 overflow-hidden">
        <div className="w-full max-w-3xl flex flex-col h-full">
          {/* Messages area */}
          <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500 py-12">
                <p className="text-sm">Start a conversation by asking a question</p>
                <p className="text-xs mt-2">Example: "What are the grading criteria for Operating Systems?"</p>
              </div>
            ) : (
              messages.map((message, idx) => (
                <div
                  key={`msg-${message.timestamp}-${idx}`}
                  className={`flex ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl p-4 ${
                      message.role === "user"
                        ? "bg-indigo-600 text-white"
                        : "bg-[#18181f] border border-white/10 text-gray-100"
                    }`}
                  >
                    <p className="leading-relaxed whitespace-pre-wrap">
                      {message.content ? (
                        parseMarkdown(message.content)
                      ) : loading && message.role === "assistant" ? (
                        <span className="text-gray-400 italic">Thinking...</span>
                      ) : (
                        ""
                      )}
                    </p>
                    {message.role === "assistant" && message.sources && message.sources.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-white/10 space-y-2">
                        <h4 className="text-xs uppercase tracking-wider text-gray-400 mb-2">
                          Sources
                        </h4>
                        {message.sources.slice(0, 3).map((source, sourceIdx) => (
                          <div
                            key={`${idx}-${source.chunk_id}-${sourceIdx}`}
                            className="rounded-lg bg-white/5 p-2 border border-white/10"
                          >
                            <div className="flex justify-between text-xs text-gray-400 mb-1">
                              <span>{source.course ?? "N/A"}</span>
                              <span>{(source.confidence * 100).toFixed(0)}%</span>
                            </div>
                            <p className="text-xs text-gray-300 line-clamp-2">
                              {source.text}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div className="bg-[#1f1f27] rounded-2xl border border-white/10 p-5 shadow-2xl shadow-indigo-900/20">
            <div className="flex flex-col gap-3 mb-3 sm:flex-row sm:items-center sm:justify-between">
              <h1 className="text-lg font-semibold">Ask the syllabus</h1>
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
                {courses.length > 0 && (
                  <select
                    value={course ?? ""}
                    onChange={(e) =>
                      setCourse(e.target.value ? e.target.value : null)
                    }
                    className="bg-[#18181f] border border-white/20 rounded-full px-3 py-1 text-xs text-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="">All courses</option>
                    {courses.map((c) => (
                      <option key={c} value={c}>
                        {formatCourseName(c)}
                      </option>
                    ))}
                  </select>
                )}
                {loading && (
                  <span className="text-xs text-indigo-300 animate-pulse">
                    Thinking...
                  </span>
                )}
              </div>
            </div>

            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (canSubmit) {
                    askQuestion();
                  }
                }
              }}
              placeholder="Example: What are the grading criteria for Operating Systems?"
              className="w-full bg-transparent text-white resize-none outline-none text-base placeholder:text-gray-500"
              rows={3}
            />

            <div className="flex justify-end mt-4">
              <button
                onClick={askQuestion}
                disabled={!canSubmit}
                className="px-5 py-2 rounded-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition flex items-center gap-2 text-sm"
              >
                {loading ? "Sending..." : "Send"}
                <span className="text-lg">➤</span>
              </button>
            </div>

            {error && (
              <p className="text-sm text-rose-300 bg-rose-950/30 border border-rose-900/40 rounded-lg px-3 py-2 mt-3">
                {error}
              </p>
            )}
          </div>
        </div>
      </main>

      <footer className="text-center text-xs text-gray-500 py-4 border-t border-white/5">
        © {new Date().getFullYear()} CSE Course RAG
      </footer>
    </div>
  );
}

export default App;
