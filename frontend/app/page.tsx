"use client";

import { useState } from "react";

export default function Home() {
  const [pgn, setPgn] = useState("");
  const [color, setColor] = useState("White");
  const [elo, setElo] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function getElo(pgn: string, color: string) {
    const response = await fetch(
      `http://localhost:5001/api/guess_elo?pgn=${encodeURIComponent(
        pgn
      )}&color=${encodeURIComponent(color)}`
    );

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Request failed");
    }

    return await response.json();
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    if (!pgn.trim()) return;

    setLoading(true);
    setError(null);
    setElo(null);

    try {
      const data = await getElo(pgn, color);
      setElo(data.elo);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Something went wrong"
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-zinc-50 flex flex-col items-center justify-center px-6">

      <div className="w-full max-w-2xl flex flex-col items-center">

        {/* Title */}
        <h1 className="text-6xl font-bold tracking-tight text-zinc-900 mb-3">
          GuessTheElo
        </h1>

        <p className="text-zinc-500 text-lg mb-10">
          Paste a chess game and see what Elo we predict.
        </p>

        {/* Form */}
        <form
          onSubmit={handleSubmit}
          className="w-full flex flex-col gap-4"
        >

          {/* PGN Input */}
          <textarea
            value={pgn}
            onChange={(e) => setPgn(e.target.value)}
            placeholder={`Paste PGN here...

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6`}
            className="
              w-full
              h-48
              rounded-xl
              border
              border-zinc-300
              bg-white
              p-5
              font-mono
              text-sm
              text-zinc-800
              shadow-sm
              outline-none
              resize-none
              focus:ring-2
              focus:ring-zinc-900
              focus:border-transparent
            "
          />

          {/* Color Selection */}
          <div className="flex gap-3">

            <button
              type="button"
              onClick={() => setColor("White")}
              className={`flex-1 rounded-lg py-3 font-medium transition
                ${
                  color === "White"
                    ? "bg-zinc-900 text-white"
                    : "bg-white border border-zinc-300 text-zinc-700"
                }
              `}
            >
              Playing White
            </button>

            <button
              type="button"
              onClick={() => setColor("Black")}
              className={`flex-1 rounded-lg py-3 font-medium transition
                ${
                  color === "Black"
                    ? "bg-zinc-900 text-white"
                    : "bg-white border border-zinc-300 text-zinc-700"
                }
              `}
            >
              Playing Black
            </button>

          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={loading || !pgn.trim()}
            className="
              w-full
              rounded-xl
              bg-zinc-900
              py-4
              text-lg
              font-semibold
              text-white
              transition
              hover:bg-zinc-700
              disabled:cursor-not-allowed
              disabled:opacity-50
            "
          >
            {loading ? "Analyzing Game..." : "Guess My Elo"}
          </button>

        </form>

        {/* Loading Spinner */}
        {loading && (
          <div className="mt-10 flex flex-col items-center gap-4">

            <div
              className="
                h-10
                w-10
                animate-spin
                rounded-full
                border-4
                border-zinc-200
                border-t-zinc-900
              "
            />

            <p className="text-zinc-500">
              Stockfish is analyzing your moves...
            </p>

          </div>
        )}

        {/* Elo Result */}
        {!loading && elo !== null && (
          <div className="mt-10 text-center">

            <p className="text-zinc-500 text-sm uppercase tracking-widest mb-2">
              Predicted Elo
            </p>

            <div className="text-7xl font-bold text-zinc-900">
              {Math.round(elo)}
            </div>

          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-8 text-red-500 text-center">
            {error}
          </div>
        )}

      </div>

    </main>
  );
}