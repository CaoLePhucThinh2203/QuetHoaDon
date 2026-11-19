// Calls a function of the Gradio app on the Hugging Face Space (Gradio HTTP API: POST, then read the event stream).
// Files starting with "_" are helpers, not Vercel routes.
async function callGradio(name, data) {
  const base = (process.env.SPACE_URL || "").replace(/\/$/, "");
  const auth = process.env.HF_TOKEN ? { Authorization: `Bearer ${process.env.HF_TOKEN}` } : {};
  const start = await fetch(`${base}/gradio_api/call/${name}`, {
    method: "POST", headers: { "Content-Type": "application/json", ...auth }, body: JSON.stringify({ data }),
  });
  if (!start.ok) throw new Error(`Space answered ${start.status} (still starting?)`);
  const { event_id } = await start.json();
  const stream = await fetch(`${base}/gradio_api/call/${name}/${event_id}`, { headers: auth });
  const text = await stream.text();               // the stream ends once the function has returned
  const m = text.match(/event: (complete|error)\s*\ndata: (.*)/);
  if (!m || m[1] === "error") throw new Error("the Space function failed" + (m && m[2] !== "null" ? `: ${m[2]}` : ""));
  return JSON.parse(m[2])[0];
}

module.exports = { callGradio };
