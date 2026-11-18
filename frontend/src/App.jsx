import { useState, useCallback, useEffect } from "react";
import { useSearch } from "./hooks/useSearch.js";
import SearchBar from "./components/SearchBar.jsx";
import PhotoCard from "./components/PhotoCard.jsx";
import Lightbox from "./components/Lightbox.jsx";
import Camera from "./components/Camera.jsx";

export default function App() {
  const { query, results, total, loading, error, search, loadMore, searchImage } = useSearch();
  const [selected, setSelected] = useState(null);
  const closeBox = useCallback(() => setSelected(null), []);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [apiUnavailable, setApiUnavailable] = useState(false);

  let status = "";
  if (error) status = "Error: " + error;
  else if (loading && results.length === 0) status = "Searching...";
  else if (query) status = `Showing ${results.length} of ${total} for "${query}"`;

  const closeCamera = useCallback(() => {
    setCameraOpen(false);
  }, []);

  async function checkApiConnection() {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE}/health`, {
        method: "GET",
        mode: "cors"
      });

      return response.ok;
    } catch (error) {
      console.error("API connection failed:", error);
      return false;
    }
  }

  useEffect(() => {
      if (cameraError) {
        alert("Unable to access camera: " + cameraError);
        setCameraError(null);
      }
    }, [cameraError]);

    useEffect(() => {
    checkApiConnection().then(connected => {
      setApiUnavailable(!connected);
    });
  }, []);

  return (
    <>
      {apiUnavailable ? (
        <div className="error">
          <p>Unable to connect to the API.</p>
          <p>
            Your browser may need you to approve the server certificate
            before using this app.
          </p>

          <a
            href={`${import.meta.env.VITE_API_BASE}/health`}
            target="_blank"
            rel="noreferrer"
          >
            Open API and approve certificate
          </a>

          <p>After accepting the certificate, return here and refresh the page.</p>
        </div>
      ) : (
        <>
          <SearchBar
            onSearch={search}
            onOpenCamera={() => setCameraOpen(true)}
          />

          {cameraOpen && (
            <Camera
              onImageSearch={searchImage}
              onClose={closeCamera}
              onError={setCameraError}
            />
          )}

          <div id="status">{status}</div>

          <div id="grid">
            {results.map(r => (
              <PhotoCard
                key={r.id}
                photo={r}
                onSelect={setSelected}
              />
            ))}
          </div>

          {results.length < total && (
            <button id="more" onClick={loadMore} disabled={loading}>
              Load more
            </button>
          )}

          <Lightbox photo={selected} query={query} onClose={closeBox} />
        </>
      )}
    </>
  );
}
