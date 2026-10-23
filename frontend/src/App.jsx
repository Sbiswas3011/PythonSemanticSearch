import { useState, useCallback } from "react";
import { useSearch } from "./hooks/useSearch.js";
import SearchBar from "./components/SearchBar.jsx";
import PhotoCard from "./components/PhotoCard.jsx";
import Lightbox from "./components/Lightbox.jsx";

export default function App() {
  const { query, results, total, loading, error, search, loadMore } = useSearch();
  const [selected, setSelected] = useState(null);
  const closeBox = useCallback(() => setSelected(null), []);

  let status = "";
  if (error) status = "Error: " + error;
  else if (loading && results.length === 0) status = "Searching...";
  else if (query) status = `Showing ${results.length} of ${total} for "${query}"`;

  return (
    <>
      <SearchBar onSearch={search} />
      <div id="status">{status}</div>

      <div id="grid">
        {results.map(r => (
          <PhotoCard key={r.id} photo={r} onSelect={setSelected} />
        ))}
      </div>

      {results.length < total && (
        <button id="more" onClick={loadMore} disabled={loading}>
          Load more
        </button>
      )}

      <Lightbox photo={selected} query={query} onClose={closeBox} />
    </>
  );
}
