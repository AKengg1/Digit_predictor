import { useEffect, useRef, useState } from "react";

const API_URL = "https://digit-predictor-0nyz.onrender.com/predict"||"http://localhost:8000/predict";
const SIZE = 280; // drawing canvas is 280x280 (28 x 10)
const BRUSH = 22; // stroke width in canvas pixels
const INK_THRESHOLD = 20; // red value above this counts as "chalk"

/* ------------------------------------------------------------------ */
/* Canvas -> 784 numbers, prepared the way MNIST digits were prepared  */
/* (unchanged: the model depends on this exact preprocessing)          */
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
  const resultRef = useRef(null);
  const drawing = useRef(false);
  const last = useRef({ x: 0, y: 0 });

  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [message, setMessage] = useState("");
  const [runs, setRuns] = useState(0); // bumps on every answer so the chalk animation replays

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

  // On phones the answer sits below the fold: bring it into view
  useEffect(() => {
    if (status === "done" || status === "error") {
      resultRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [status]);

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
    const ctx = canvasRef.current.getContext("2d");
    // Touch screens batch their samples: using them all keeps fast strokes smooth
    const batch = e.nativeEvent.getCoalescedEvents?.();
    const samples = batch && batch.length ? batch : [e];
    ctx.beginPath();
    ctx.moveTo(last.current.x, last.current.y);
    for (const s of samples) {
      const p = point(s);
      ctx.lineTo(p.x, p.y);
      last.current = p;
    }
    ctx.stroke();
  }

  function stopDrawing() {
    drawing.current = false;
  }

  function clearBoard() {
    resetBoard();
    setResult(null);
    setMessage("");
    setStatus("idle");
    canvasRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  async function recognize() {
    const pixels = canvasToPixels(canvasRef.current);
    if (!pixels) {
      setStatus("error");
      setMessage("The board is empty. Draw a digit, then tap Recognize digit.");
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
        "Can’t reach the model. Check your connection and try again. Running locally? Start the API with “uvicorn main:app --reload”."
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
    setRuns((n) => n + 1);
    setStatus("done");
  }

  const done = status === "done" && result;
  const confidence = result ? Math.round(result.probabilities[result.digit] * 100) : 0;

  return (
    <div className="dr-app">
      <style>{css}</style>

      <div className="dr-frame">
        <div className="dr-board">
          <header className="dr-header">
            <h1>Digit Reader</h1>
            <svg className="dr-scribble" viewBox="0 0 150 10" preserveAspectRatio="none" aria-hidden="true">
              <path d="M2 6 C 25 1, 40 9, 70 5 S 115 2, 148 6" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
            </svg>
            <p>
              Draw one digit from 0 to 9. A neural network built from scratch in NumPy reads it.
            </p>
          </header>

          <main className="dr-layout">
            <section className="dr-pad" aria-label="Drawing board">
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
                onContextMenu={(e) => e.preventDefault()}
              />
            </section>

            <section ref={resultRef} className="dr-result" aria-label="Result" aria-live="polite">
              {status === "error" ? (
                <p className="dr-error">{message}</p>
              ) : (
                <div className="dr-verdict">
                  <span key={runs} className={"dr-digit" + (done ? " is-on" : "")}>
                    {done ? result.digit : "?"}
                  </span>
                  {done ? (
                    <p className="dr-sure">
                      <strong>{confidence}%</strong> sure
                    </p>
                  ) : (
                    <p className="dr-hint">
                      {status === "loading"
                        ? "Reading your digit…"
                        : "Draw a digit on the board, then tap Recognize digit."}
                    </p>
                  )}
                </div>
              )}

              <ul className="dr-bars" key={runs} aria-label="Probability for each digit">
                {Array.from({ length: 10 }, (_, d) => {
                  const p = done ? result.probabilities[d] : 0;
                  const top = done && d === result.digit;
                  return (
                    <li key={d} className={"dr-col" + (top ? " is-top" : "")} style={{ "--i": d }}>
                      <span className="dr-label">{d}</span>
                      <span className="dr-pct">{done ? (p < 0.01 ? "<1" : Math.round(p * 100)) : ""}</span>
                      <span className="dr-track">
                        <span className="dr-fill" style={{ height: `${(p * 100).toFixed(1)}%` }} />
                      </span>
                    </li>
                  );
                })}
              </ul>
            </section>
          </main>

          <div className="dr-tray">
            <button className="dr-btn dr-eraser" onClick={clearBoard}>
              Clear
            </button>
            <button
              className="dr-btn dr-chalk"
              onClick={recognize}
              disabled={status === "loading"}
            >
              Recognize digit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Styles (scoped with the dr- prefix so nothing else is affected)     */
/* ------------------------------------------------------------------ */
const css = `
@import url("https://fonts.googleapis.com/css2?family=Patrick+Hand&family=Source+Sans+3:wght@400;600;700&display=swap");

.dr-app {
  --wall: #d5dfdb;
  --chalk: #f4f3ea;
  --dim: #a9b9b3;
  --gold: #f5d76e;
  --sky: #8ec5e8;
  --error: #f4a8a0;
  --oak: #a47a4f;
  --oak-hi: #c9a272;
  --oak-lo: #6c4c2e;

  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  min-height: 100vh;
  min-height: 100dvh;
  padding: max(8px, env(safe-area-inset-top)) 8px max(8px, env(safe-area-inset-bottom));
  background: var(--wall);
  color: var(--chalk);
  font-family: "Source Sans 3", system-ui, sans-serif;
  text-align: left;
  -webkit-text-size-adjust: 100%;
}
.dr-app *, .dr-app *::before, .dr-app *::after { box-sizing: border-box; }

/* Wooden frame + slate board. On phones the board fills the screen. */
.dr-frame {
  display: flex;
  flex: 1 0 auto;
  width: 100%;
  max-width: 940px;
  padding: 7px;
  border-radius: 18px;
  background: linear-gradient(135deg, var(--oak-hi), var(--oak) 40%, var(--oak-lo));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.35), 0 20px 40px -20px rgba(18, 32, 29, 0.55);
}
.dr-board {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
  padding: 18px 14px 0;
  border-radius: 12px;
  background:
    radial-gradient(60% 40% at 18% 14%, rgba(255, 255, 255, 0.05), transparent 70%),
    radial-gradient(50% 35% at 82% 68%, rgba(255, 255, 255, 0.04), transparent 70%),
    radial-gradient(120% 90% at 50% 0%, #2b4b44 0%, #1d342f 75%);
  box-shadow: inset 0 0 0 2px rgba(0, 0, 0, 0.25), inset 0 10px 30px rgba(0, 0, 0, 0.35);
}

.dr-header h1 {
  margin: 0;
  font: 400 clamp(40px, 12vw, 54px) / 1 "Patrick Hand", "Comic Sans MS", cursive;
  text-shadow: 0 0 14px rgba(244, 243, 234, 0.25);
}
.dr-scribble { display: block; width: 150px; height: 10px; margin: 4px 0 10px; color: var(--gold); }
.dr-header p { margin: 0; max-width: 46ch; font-size: 16px; line-height: 1.5; color: var(--dim); }

.dr-layout {
  display: grid;
  flex: 1;
  grid-template-columns: minmax(0, 1fr);
  align-content: start;
  gap: 22px;
  margin-top: 20px;
}

/* The pad: you write straight onto the slate (the black canvas is screen-blended away) */
.dr-pad {
  padding: 8px;
  border: 2px dashed rgba(244, 243, 234, 0.3);
  border-radius: 22px;
  box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.22);
}
.dr-canvas {
  display: block;
  width: 100%;
  height: auto;
  aspect-ratio: 1;
  border-radius: 14px;
  background: #000;
  mix-blend-mode: screen;
  cursor: crosshair;
  touch-action: none;
  -webkit-user-select: none;
  user-select: none;
  -webkit-touch-callout: none;
}

/* Result */
.dr-result { scroll-margin-bottom: calc(110px + env(safe-area-inset-bottom)); }
.dr-verdict { display: flex; align-items: center; gap: 16px; min-height: 104px; }
.dr-digit {
  min-width: 0.62em;
  font: 400 112px / 1 "Patrick Hand", "Comic Sans MS", cursive;
  text-align: center;
  color: rgba(244, 243, 234, 0.2);
}
.dr-digit.is-on {
  color: var(--gold);
  text-shadow: 0 0 22px rgba(245, 215, 110, 0.38);
  animation: dr-pop 0.45s cubic-bezier(0.2, 1.5, 0.4, 1);
}
@keyframes dr-pop { from { transform: scale(0.55) rotate(-6deg); opacity: 0; } }

.dr-hint, .dr-error, .dr-sure {
  margin: 0;
  font-family: "Patrick Hand", "Comic Sans MS", cursive;
  line-height: 1.3;
}
.dr-hint { font-size: 24px; color: var(--dim); }
.dr-error { font-size: 22px; color: var(--error); min-height: 104px; }
.dr-sure { font-size: 28px; color: var(--dim); }
.dr-sure strong { display: block; font-weight: 400; font-size: 46px; line-height: 1.1; color: var(--chalk); }

/* Chalk bar chart: one column per digit */
.dr-bars {
  display: grid;
  grid-template-columns: repeat(10, minmax(0, 1fr));
  gap: 6px;
  height: 176px;
  margin: 16px 0 0;
  padding: 0;
  list-style: none;
}
.dr-col { display: grid; grid-template-rows: auto 1fr auto; gap: 4px; justify-items: center; min-width: 0; }
.dr-pct { order: 1; min-height: 15px; font: 600 12px / 15px "Source Sans 3", system-ui, sans-serif; font-variant-numeric: tabular-nums; color: var(--dim); }
.dr-track { order: 2; position: relative; width: 100%; border-radius: 6px; background: rgba(244, 243, 234, 0.08); overflow: hidden; }
.dr-label { order: 3; font: 400 24px / 1 "Patrick Hand", "Comic Sans MS", cursive; color: var(--dim); }
.dr-fill {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  border-radius: 6px;
  background: var(--chalk);
  transform-origin: bottom;
  animation: dr-grow 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) both;
  animation-delay: calc(var(--i) * 40ms);
}
@keyframes dr-grow { from { transform: scaleY(0); } }
.dr-col.is-top .dr-fill { background: var(--gold); box-shadow: 0 0 16px rgba(245, 215, 110, 0.5); }
.dr-col.is-top .dr-label, .dr-col.is-top .dr-pct { color: var(--gold); }

/* The chalk ledge: sticks to the bottom of the screen, inside thumb reach */
.dr-tray {
  position: sticky;
  bottom: 0;
  z-index: 5;
  display: flex;
  justify-content: center;
  gap: 12px;
  margin: 22px -14px 0;
  padding: 14px 14px calc(14px + env(safe-area-inset-bottom));
  border-top: 3px solid var(--oak-lo);
  border-radius: 0 0 12px 12px;
  background: linear-gradient(var(--oak-hi), var(--oak) 55%, #8f6a45);
  box-shadow: 0 -8px 20px rgba(0, 0, 0, 0.25);
}
.dr-btn {
  position: relative;
  display: grid;
  place-items: center;
  min-height: 56px;
  border: 0;
  font: 700 18px "Source Sans 3", system-ui, sans-serif;
  cursor: pointer;
  -webkit-tap-highlight-color: transparent;
  transition: transform 0.1s, box-shadow 0.1s;
}
.dr-btn:active:not(:disabled) { transform: translateY(3px); }
.dr-btn:focus-visible { outline: 3px solid var(--sky); outline-offset: 3px; }
.dr-btn:disabled { opacity: 0.6; cursor: wait; }

.dr-eraser {
  flex: 1 1 0;
  max-width: 200px;
  padding: 20px 12px 0;
  border-radius: 10px;
  color: #4a3622;
  background: linear-gradient(#3b4442 0 20px, #e2d5b8 20px);
  box-shadow: 0 5px 0 #8a7656, 0 10px 14px rgba(0, 0, 0, 0.3);
}
.dr-eraser:active { box-shadow: 0 2px 0 #8a7656, 0 4px 8px rgba(0, 0, 0, 0.3); }
.dr-chalk {
  flex: 2 1 0;
  max-width: 340px;
  padding: 0 16px;
  border-radius: 14px;
  color: #1b2f2b;
  background: linear-gradient(#fcfbf4, #e4e2d3);
  box-shadow: 0 5px 0 #a9a798, 0 10px 14px rgba(0, 0, 0, 0.3);
}
.dr-chalk:active:not(:disabled) { box-shadow: 0 2px 0 #a9a798, 0 4px 8px rgba(0, 0, 0, 0.3); }

/* Tablets and desktops: board hugs its content, two columns */
@media (min-width: 820px) {
  .dr-app { padding: 40px 24px 56px; }
  .dr-frame { flex: 0 0 auto; padding: 12px; border-radius: 22px; }
  .dr-board { padding: 32px 32px 0; border-radius: 14px; }
  .dr-layout { grid-template-columns: minmax(0, 420px) minmax(0, 1fr); align-items: start; gap: 36px; margin-top: 28px; }
  .dr-bars { height: 230px; gap: 8px; }
  .dr-tray { position: static; margin: 32px -32px 0; padding: 16px 32px; border-radius: 0 0 14px 14px; }
}

@media (prefers-reduced-motion: reduce) {
  .dr-app *, .dr-app *::before, .dr-app *::after { animation: none !important; transition: none !important; }
}
`;