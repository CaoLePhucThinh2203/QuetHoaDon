// Vercel function: asks the Space whether the model is loaded (also wakes a sleeping Space).
const { callGradio } = require("./_gradio");

module.exports = async (req, res) => {
  try {
    return res.status(200).json(await callGradio("health", []));
  } catch (e) {
    return res.status(200).json({ status: "starting", error: e.message });
  }
};
