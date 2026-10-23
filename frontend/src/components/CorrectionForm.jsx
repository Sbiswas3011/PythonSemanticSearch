import { useState } from "react";
import { correctEmbedding } from "../api";

export default function CorrectionForm({ photo, query }) {
  const [correct, setCorrect] = useState("");
  const [params, setParams] = useState({
    margin: 0.15,
    preserveWeight: 2.0,
    lr: 0.01,
    steps: 200
  });
  const [status, setStatus] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const setParam = key => e => setParams(p => ({ ...p, [key]: e.target.value }));

  async function handleSubmit(e) {
    e.preventDefault();

    if (!Number.isInteger(Number(photo.index))) {
      setStatus("Missing image embedding index.");
      return;
    }

    setSubmitting(true);
    setStatus("Updating embedding...");
    try {
      await correctEmbedding({
        index: Number(photo.index),
        wrong: query,
        correct: correct.trim(),
        margin: Number(params.margin),
        preserve_weight: Number(params.preserveWeight),
        lr: Number(params.lr),
        steps: Number(params.steps)
      });
      setStatus("Correction applied successfully.");
    } catch (err) {
      setStatus("Error: " + err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form id="correction-form" onSubmit={handleSubmit}>
      <h3>Correct embedding</h3>

      <label>
        Correct label
        <input
          value={correct}
          onChange={e => setCorrect(e.target.value)}
          placeholder="e.g. blue-eared kingfisher"
          required
        />
      </label>

      <div className="params">
        <label>
          Margin
          <input type="number" step="0.01" min="0.1" required
                 value={params.margin} onChange={setParam("margin")} />
        </label>
        <label>
          Preserve weight
          <input type="number" step="0.1" min="1.0" required
                 value={params.preserveWeight} onChange={setParam("preserveWeight")} />
        </label>
        <label>
          Learning rate
          <input type="number" step="0.01" min="0.01" required
                 value={params.lr} onChange={setParam("lr")} />
        </label>
        <label>
          Steps
          <input type="number" step="1" min="1" required
                 value={params.steps} onChange={setParam("steps")} />
        </label>
      </div>

      <button type="submit" disabled={submitting}>Apply correction</button>
      <div id="correction-status">{status}</div>
    </form>
  );
}
