// Vercel function: forwards a receipt photo to the Gradio app on the Hugging Face Space and returns its JSON.
// The browser sends {"image": "<base64 JPEG>"}; the tokens stay here (environment variables):
//   SPACE_URL  https://<user>-<space>.hf.space        (required)
//   HF_TOKEN   token that can read a private Space     (optional)
//   API_KEY    same value as the Space secret API_KEY  (optional)
const { callGradio } = require("./_gradio");

module.exports = async (req, res) => {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const { image } = req.body || {};
  if (!image) return res.status(400).json({ error: "no image" });
  if (image.length > 6 * 1024 * 1024) return res.status(413).json({ error: "image larger than 4 MB" });
  try {
    const out = await callGradio("extract_b64", [image, process.env.API_KEY || ""]);
    if (out && out.error) return res.status(String(out.error).startsWith("model") ? 503 : 400).json(out);
    return res.status(200).json(out);
  } catch (e) {
    return res.status(502).json({ error: e.message });
  }
};
