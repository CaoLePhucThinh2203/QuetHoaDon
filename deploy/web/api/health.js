// Vercel function: asks the Space whether the model is loaded (also wakes a sleeping Space).
module.exports = async (req, res) => {
  const headers = {};
  if (process.env.HF_TOKEN) headers.Authorization = `Bearer ${process.env.HF_TOKEN}`;
  try {
    const r = await fetch(`${process.env.SPACE_URL}/health`, { headers });
    const text = await r.text();
    try { return res.status(200).json(JSON.parse(text)); } catch { return res.status(200).json({ status: "starting" }); }
  } catch (e) {
    return res.status(200).json({ status: "starting", error: e.message });
  }
};
