// Vercel function: forwards a receipt photo to the Hugging Face Space and returns its JSON.
// The browser sends {"image": "<base64 JPEG>", "name": "..."}; the tokens stay here (environment variables):
//   SPACE_URL  https://<user>-<space>.hf.space        (required)
//   HF_TOKEN   token that can read a private Space     (optional)
//   API_KEY    same value as the Space secret API_KEY  (optional)
module.exports = async (req, res) => {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const { image, name } = req.body || {};
  if (!image) return res.status(400).json({ error: "no image" });
  const bytes = Buffer.from(image, "base64");
  if (bytes.length > 4 * 1024 * 1024) return res.status(413).json({ error: "image larger than 4 MB" });

  const form = new FormData();
  form.append("file", new Blob([bytes], { type: "image/jpeg" }), (name || "receipt") + ".jpg");
  const headers = {};
  if (process.env.HF_TOKEN) headers.Authorization = `Bearer ${process.env.HF_TOKEN}`;
  if (process.env.API_KEY) headers["x-api-key"] = process.env.API_KEY;
  try {
    const r = await fetch(`${process.env.SPACE_URL}/extract`, { method: "POST", body: form, headers });
    const text = await r.text();
    let body;
    try { body = JSON.parse(text); } catch { body = { error: `Space answered ${r.status} (still starting?)` }; }
    return res.status(r.status).json(body);
  } catch (e) {
    return res.status(502).json({ error: `cannot reach the Space: ${e.message}` });
  }
};
