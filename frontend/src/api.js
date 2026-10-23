const API_BASE = import.meta.env.VITE_API_BASE;

export const apiUrl = path => `${API_BASE}${path}`;
export const imgUrl = id => apiUrl("/img/" + encodeURIComponent(id));

export async function searchPhotos(q, offset) {
  const res = await fetch(
    apiUrl("/search?q=" + encodeURIComponent(q) + "&offset=" + offset)
  );
  if (!res.ok) throw new Error("Search failed (" + res.status + ")");
  return res.json();
}

export async function correctEmbedding(payload) {
  const res = await fetch(apiUrl("/correct"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Correction failed");
  return data;
}
