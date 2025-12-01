import { useCallback, useEffect, useMemo, useState } from "react";

type SourceChunk = {
  chunk_id: string;
  text: string;
  score: number;
  confidence: number;
  course?: string | null;
  metadata?: Record<string, unknown> | null;
};

/* const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api"; */

const API_BASE_URL = "http://localhost:8000/api";

// Derive API root (backend base URL without /api)
const API_ROOT = API_BASE_URL.replace(/\/api\/?$/, "");

function formatCourseName(courseId: string): string {
  // Simple formatting: replace underscores with spaces
  return courseId.replace(/_/g, " ");
}

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<SourceChunk[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [course, setCourse] = useState<string | null>(null);
  const [courses, setCourses] = useState<string[]>([]);

  const canSubmit = useMemo(
    () => question.trim().length > 0 && !loading,
    [question, loading]
  );

  const askQuestion = useCallback(async () => {
    if (!canSubmit) {
      return;
    }

    setLoading(true);
    setError(null);
    setAnswer(null);
    setSources([]);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          course: course,
          top_k: 5
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

      setAnswer(data.answer);
      setSources(data.sources ?? []);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to reach backend.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [canSubmit, question, course]);

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
        <div className="text-xs text-gray-400">
          Powered by Retrieval-Augmented Generation
        </div>
      </div>

      <main className="flex-1 flex flex-col items-center px-4 py-10 gap-6">
        <div className="w-full max-w-3xl space-y-4">
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
                    void askQuestion();
                  }
                }
              }}
              placeholder="Example: What are the grading criteria for Operating Systems?"
              className="w-full bg-transparent text-white resize-none outline-none text-base placeholder:text-gray-500"
              rows={4}
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

          {answer && (
            <section className="bg-[#18181f] rounded-2xl border border-white/5 p-5 space-y-4">
              <div>
                <h2 className="text-sm uppercase tracking-wider text-gray-400">
                  Answer
                </h2>
                <p className="mt-2 leading-relaxed text-gray-100">{answer}</p>
              </div>

              {sources.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-xs uppercase tracking-wider text-gray-500">
                    Sources
                  </h3>
                  <div className="space-y-2">
                    {sources.map((source) => (
                      <div
                        key={source.chunk_id}
                        className="rounded-xl bg-white/5 p-3 border border-white/10"
                      >
                        <div className="flex justify-between text-xs text-gray-400">
                          <span>{source.course ?? "N/A"}</span>
                          <span>{(source.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <p className="mt-2 text-sm text-gray-200 overflow-hidden">
                          {source.text}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}
        </div>
      </main>

      <footer className="text-center text-xs text-gray-500 py-4 border-t border-white/5">
        © {new Date().getFullYear()} CSE Course RAG
      </footer>
    </div>
  );
}

export default App;
