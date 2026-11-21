import { useState, useCallback, useEffect, useRef } from "react";
import { useSearch } from "./hooks/useSearch.js";
import SearchBar from "./components/SearchBar.jsx";
import PhotoCard from "./components/PhotoCard.jsx";
import Lightbox from "./components/Lightbox.jsx";
import Camera from "./components/Camera.jsx";

export default function App() {
  const {
    query,
    results,
    total,
    loading,
    error,
    search,
    loadMore,
    searchImage,
  } = useSearch();
  const [selected, setSelected] = useState(null);
  const closeBox = useCallback(() => setSelected(null), []);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [apiUnavailable, setApiUnavailable] = useState(false);
  const [dim, setDim] = useState(0);
  const cameraRef = useRef(null);
  const [quality, setQuality] = useState("original");

  let status = "";
  if (error) status = "Error: " + error;
  else if (loading && results.length === 0) status = "Searching...";
  else if (query)
    status = `Showing ${results.length} of ${total} for "${query}"`;

  const closeCamera = useCallback(() => {
    setCameraOpen(false);
  }, []);

  async function checkApiConnection() {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE}/health`, {
        method: "GET",
        mode: "cors",
        signal: AbortSignal.timeout(1000),
      });

      return response.ok;
    } catch (error) {
      console.error("API connection failed:", error);
      return false;
    }
  }

  useEffect(() => {
    if (cameraOpen) {
      // cameraRef.current?.scrollIntoView({
      //   behavior: "smooth",
      //   block: "start",
      // });
      window.scrollTo({ top: 0, behavior: "smooth" });

      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }

    return () => {
      document.body.style.overflow = "";
    };
  }, [cameraOpen]);

  useEffect(() => {
    if (cameraError) {
      alert("Unable to access camera: " + cameraError);
      setCameraError(null);
    }
  }, [cameraError]);

  useEffect(() => {
    let active = true;

    const check = async () => {
      const connected = await checkApiConnection();

      if (active) {
        setApiUnavailable(!connected);
      }
    };

    check();

    const interval = setInterval(check, 2000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <>
      {apiUnavailable ? (
        <div className="error">
          <p>Unable to connect to the API.</p>
          <p>
            Your browser may need you to approve the server certificate before
            using this app.
          </p>

          <a
            href={`${import.meta.env.VITE_API_BASE}/health`}
            target="_blank"
            rel="noreferrer"
          >
            Open API and approve certificate
          </a>

          <p>
            After accepting the certificate, return here and refresh the page.
          </p>
        </div>
      ) : (
        <>
          <SearchBar
            onSearch={search}
            onImageSearch={searchImage}
            onOpenCamera={() => setCameraOpen(true)}
            dim={dim}
            setDim={setDim}
          />

          {cameraOpen && (
            <div ref={cameraRef}>
              <Camera
                onImageSearch={searchImage}
                onClose={closeCamera}
                onError={setCameraError}
                dim={dim}
                quality={quality}
                setQuality={setQuality}
              />
            </div>
          )}

          <div id="status">{status}</div>

          <div id="grid">
            {results.map((r) => (
              <PhotoCard key={r.id} photo={r} onSelect={setSelected} />
            ))}
          </div>

          {results.length < total && (
            <button id="more" onClick={() => loadMore(dim)} disabled={loading}>
              Load more
            </button>
          )}

          <Lightbox
            photo={selected}
            query={query}
            onClose={closeBox}
            dim={dim}
          />
        </>
      )}
    </>
  );
}
