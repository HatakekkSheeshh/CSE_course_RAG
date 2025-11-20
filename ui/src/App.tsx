import { useState } from "react";

function App() {
  const [text, setText] = useState("");

  return (
    <div className="min-h-screen bg-[#3a3a3a] flex flex-col">
      {/* Top bar */}
      <div className="flex items-center justify-between px-6 py-4 text-white text-sm">
        {/* <div className="font-semibold tracking-wide">HuuCau</div> */}

        {/* <div className="flex gap-6 opacity-80">
          <span>A</span>
          <span>B</span>
          <span>C</span>
          <span>D</span>
          <span>E</span>
        </div> */}

        <div className="ml-auto flex gap-2">
          <button className="px-3 py-1 rounded-full bg-gray-600 text-xs">
            Sign up
          </button>
          <button className="px-3 py-1 rounded-full bg-indigo-500 text-xs">
            Sign in
          </button>
        </div>
      </div>

      {/* Center content */}
      <div className="flex-1 flex flex-col items-center justify-center">
        <div className="text-center mb-6">
          <div className="text-5xl mb-4">❄️</div>
          <h1 className="text-white text-lg font-semibold">
            {/* Artificial Intelligence by Quoc Hieu */}
          </h1>
        </div>

        <div className="w-full max-w-xl bg-[#2f2f2f] rounded-2xl border border-gray-600 px-4 py-3">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Ask something hard."
            className="w-full bg-transparent text-white resize-none outline-none text-sm"
            rows={2}
          />

          <div className="flex justify-end mt-2">
            <button className="w-10 h-10 flex items-center justify-center rounded-full bg-indigo-600 hover:bg-indigo-500 transition">
              ➤
            </button>
          </div>
        </div>
      </div>

      <div className="text-center text-xs text-gray-400 py-4">
        © 2025 CSE_COURSE_RAG
      </div>
    </div>
  );
}

export default App;
