const API_BASE = import.meta.env.VITE_API_BASE;

export const apiUrl = path => `${API_BASE}${path}`;
export const imgUrl = id => apiUrl("/img/" + encodeURIComponent(id));

export async function searchPhotos(q, offset, dim = 0) {
  const res = await fetch(
    apiUrl("/search?q=" + encodeURIComponent(q) + "&offset=" + offset + "&dim=" + dim)
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

export async function searchPhotosByImage(file, offset = 0, size = "tiny", dim = 0) {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("offset", offset);
  formData.append("size", size);
  // formData.append("dim", dim);

  const params = new URLSearchParams({
    dim: String(dim),
  });

  const response = await fetch(`${API_BASE}/search?${params}`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error("Image search failed");
  }

  return response.json();
}

export async function searchPhotosById(searchId, offset) {
  const params = new URLSearchParams({
    search_id: searchId,
    offset: String(offset),
  });

  const response = await fetch(`${API_BASE}/search?${params}`);

  if (!response.ok) {
    throw new Error("Image search expired or failed");
  }

  return response.json();
}