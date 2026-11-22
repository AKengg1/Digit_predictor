import { useEffect, useRef, useState } from "react";

const API_URL = "http://localhost:8000/predict";
const SIZE = 280; // drawing canvas is 280x280 (28 x 10)
const BRUSH = 22; // stroke width in canvas pixels
const INK_THRESHOLD = 20; // red value above this counts as "chalk"

/* ------------------------------------------------------------------ */
/* Canvas -> 784 numbers, prepared the way MNIST digits were prepared  */
/* ------------------------------------------------------------------ */
function canvasToPixels(canvas) {
  const ctx = canvas.getContext("2d");
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);

  // 1. Bounding box of everything that was drawn
  let minX = SIZE, minY = SIZE, maxX = -1, maxY = -1;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (data[(y * SIZE + x) * 4] > INK_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null; // empty board

  // 2. Scale the digit to fit a 20x20 box (keeping its proportions),
  //    and place it in the middle of a 28x28 image
  const w = maxX - minX + 1;
  const h = maxY - minY + 1;
  const scale = 20 / Math.max(w, h);
  const tw = Math.max(1, Math.round(w * scale));
  const th = Math.max(1, Math.round(h * scale));

  const small = document.createElement("canvas");
  small.width = 28;
  small.height = 28;
  const sctx = small.getContext("2d");
  sctx.fillStyle = "#000";
  sctx.fillRect(0, 0, 28, 28);
  sctx.imageSmoothingEnabled = true;
  sctx.imageSmoothingQuality = "high";
  sctx.drawImage(
    canvas,
    minX, minY, w, h,
    Math.round((28 - tw) / 2), Math.round((28 - th) / 2), tw, th
  );

  // 3. Grayscale values 0-1 (red channel is enough: the board is black and white)
  const px = sctx.getImageData(0, 0, 28, 28).data;
  const gray = new Array(784);
  let mass = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) {
    const v = px[i * 4] / 255;
    gray[i] = v;
    mass += v;
    cx += v * (i % 28);
    cy += v * Math.floor(i / 28);
  }
  if (mass === 0) return null;
  cx /= mass;
  cy /= mass;

  // 4. Shift so the center of mass sits at the middle of the image
  const dx = Math.round(14 - cx);
  const dy = Math.round(14 - cy);
  const out = new Array(784).fill(0);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = gray[y * 28 + x];
      if (v > 0) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) out[ny * 28 + nx] = v;
      }
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */
export default function App() {
  const canvasRef = useRef(null);
  const drawing = useRef(false);
  const last = useRef({ x: 0, y: 0 });

  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [message, setMessage] = useState("");

  function resetBoard() {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = BRUSH;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }

  useEffect(() => {
    resetBoard();
  }, []);

  // Pointer position in canvas coordinates (works at any displayed size)
  function point(e) {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) * SIZE) / rect.width,
      y: ((e.clientY - rect.top) * SIZE) / rect.height,
    };
  }

  function onPointerDown(e) {
    canvasRef.current.setPointerCapture(e.pointerId);
    drawing.current = true;
    const p = point(e);
    last.current = p;
    const ctx = canvasRef.current.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(p.x + 0.01, p.y); // a tap leaves a dot
    ctx.stroke();
  }

  function onPointerMove(e) {
    if (!drawing.current) return;
    const p = point(e);
    const ctx = canvasRef.current.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(last.current.x, last.current.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    last.current = p;
  }

  function stopDrawing() {
    drawing.current = false;
  }

  function clearBoard() {
    resetBoard();
    setResult(null);
    setMessage("");
    setStatus("idle");
  }

  async function recognize() {
    const pixels = canvasToPixels(canvasRef.current);
    if (!pixels) {
      setStatus("error");
      setMessage("The board is empty. Draw a digit, then choose Recognize digit.");
      return;
    }

    setStatus("loading");
    setMessage("");

    let res;
    try {
      res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixels }),
      });
    } catch {
      setStatus("error");
      setMessage(
        "Can’t reach the model. Start the API with “uvicorn main:app --reload” in the web folder, then try again."
      );
      return;
    }

    if (!res.ok) {
      let detail = `The server answered with status ${res.status}.`;
      try {
        const body = await res.json();
        if (typeof body.detail === "string") detail = body.detail;
      } catch {
        /* keep the default message */
      }
      setStatus("error");
      setMessage(detail);
      return;
    }

    setResult(await res.json());
    setStatus("done");
  }

  const confidence = result ? Math.round(result.probabilities[result.digit] * 100) : 0;

  return (
    <div className="dr-app">
      <style>{css}</style>

      <header className="dr-header">
        <h1>Digit Reader</h1>
        <p>
          Draw one digit from 0 to 9. A neural network built from scratch in NumPy reads it.
        </p>
      </header>

      <main className="dr-layout">
        <section aria-label="Drawing board">
          <div className="dr-frame">
            <canvas
              ref={canvasRef}
              className="dr-canvas"
              width={SIZE}
              height={SIZE}
              role="img"
              aria-label="Drawing board. Draw a digit with your mouse, finger or pen."
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={stopDrawing}
              onPointerCancel={stopDrawing}
            />
          </div>
          <div className="dr-actions">
            <button
              className="dr-btn dr-btn-primary"
              onClick={recognize}
              disabled={status === "loading"}
            >
              Recognize digit
            </button>
            <button className="dr-btn dr-btn-quiet" onClick={clearBoard}>
              Clear board
            </button>
          </div>
        </section>

        <section className="dr-board" aria-label="Result" aria-live="polite">
          {status === "idle" && (
            <p className="dr-hint">
              Draw a digit on the board, then choose Recognize digit.
            </p>
          )}

          {status === "loading" && <p className="dr-hint">Reading your digit…</p>}

          {status === "error" && <p className="dr-error">{message}</p>}

          {status === "done" && result && (
            <>
              <div className="dr-verdict">
                <span className="dr-digit">{result.digit}</span>
                <span className="dr-sure">{confidence}% sure</span>
              </div>
              <ul className="dr-rows">
                {result.probabilities.map((p, digit) => (
                  <li key={digit} className="dr-row">
                    <span className="dr-label">{digit}</span>
                    <span className="dr-track">
                      <span
                        className={
                          "dr-fill" + (digit === result.digit ? " dr-fill-top" : "")
                        }
                        style={{ width: `${(p * 100).toFixed(1)}%` }}
                      />
                    </span>
                    <span className="dr-pct">{(p * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Styles (scoped with the dr- prefix so nothing else is affected)     */
/* ------------------------------------------------------------------ */
const css = `
@import url("https://fonts.googleapis.com/css2?family=Patrick+Hand&family=Source+Sans+3:wght@400;600&display=swap");

.dr-app {
  --sage: #d5dfdb;
  --ink: #1c2a29;
  --oak: #9b7b58;
  --oak-dark: #6f5237;
  --slate: #213230;
  --chalk: #f3f2ea;
  --chalk-dim: #a9b5b1;
  --sky: #8ec5e8;
  --error: #f2a9a0;

  box-sizing: border-box;
  min-height: 100vh;
  padding: 40px 20px 56px;
  background: var(--sage);
  color: var(--ink);
  font-family: "Source Sans 3", system-ui, sans-serif;
  text-align: left;
}
.dr-app *, .dr-app *::before, .dr-app *::after { box-sizing: border-box; }

.dr-header { max-width: 880px; margin: 0 auto; }
.dr-header h1 {
  margin: 0 0 8px;
  font-family: "Patrick Hand", "Comic Sans MS", cursive;
  font-size: 48px;
  font-weight: 400;
  line-height: 1.1;
}
.dr-header p { margin: 0; max-width: 52ch; font-size: 18px; line-height: 1.5; }

.dr-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 32px;
  max-width: 880px;
  margin: 32px auto 0;
}
@media (min-width: 760px) {
  .dr-layout { grid-template-columns: minmax(0, 360px) minmax(0, 1fr); align-items: start; }
}

.dr-frame {
  padding: 14px;
  background: var(--oak);
  border-radius: 6px;
  box-shadow: inset 0 0 0 2px var(--oak-dark);
}
.dr-canvas {
  display: block;
  width: 100%;
  height: auto;
  aspect-ratio: 1;
  background: #000;
  border-radius: 2px;
  cursor: crosshair;
  touch-action: none;
}

.dr-actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; }
.dr-btn {
  padding: 11px 20px;
  border: 2px solid var(--ink);
  border-radius: 6px;
  font: 600 16px "Source Sans 3", system-ui, sans-serif;
  cursor: pointer;
}
.dr-btn-primary { background: var(--ink); color: var(--chalk); }
.dr-btn-quiet { background: transparent; color: var(--ink); }
.dr-btn:disabled { opacity: 0.55; cursor: wait; }
.dr-btn:focus-visible { outline: 3px solid var(--oak-dark); outline-offset: 3px; }

.dr-board {
  min-height: 360px;
  padding: 24px;
  background: var(--slate);
  color: var(--chalk);
  border-radius: 6px;
  box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.08);
}
.dr-hint, .dr-error {
  margin: 0;
  font-family: "Patrick Hand", "Comic Sans MS", cursive;
  font-size: 26px;
  line-height: 1.35;
}
.dr-hint { color: var(--chalk-dim); }
.dr-error { color: var(--error); }

.dr-verdict { display: flex; align-items: baseline; gap: 20px; }
.dr-digit {
  font-family: "Patrick Hand", "Comic Sans MS", cursive;
  font-size: 120px;
  line-height: 1;
}
.dr-sure {
  font-family: "Patrick Hand", "Comic Sans MS", cursive;
  font-size: 28px;
  color: var(--chalk-dim);
}

.dr-rows { display: grid; gap: 8px; margin: 20px 0 0; padding: 0; list-style: none; }
.dr-row { display: grid; grid-template-columns: 24px 1fr 56px; align-items: center; gap: 12px; }
.dr-label { font-family: "Patrick Hand", "Comic Sans MS", cursive; font-size: 22px; }
.dr-track {
  display: block;
  height: 12px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  overflow: hidden;
}
.dr-fill {
  display: block;
  height: 100%;
  background: var(--chalk);
  border-radius: 6px;
  transition: width 0.4s ease;
}
.dr-fill-top { background: var(--sky); }
.dr-pct { font-size: 14px; text-align: right; font-variant-numeric: tabular-nums; color: var(--chalk-dim); }

@media (prefers-reduced-motion: reduce) {
  .dr-fill { transition: none; }
}
`;