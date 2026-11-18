import { useState, useCallback, useRef } from "react";
import { searchPhotos, searchPhotosByImage, searchPhotosById } from "../api";

export function useSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchId, setSearchId] = useState(null);
  const requestId = useRef(0); // ignore responses from outdated searches
  const imageRef = useRef(null);

  const run = useCallback(async (q, offset, dim = 0, image = null, searchId = null, size = "tiny") => {
    const id = ++requestId.current;
    setLoading(true);
    setError(null);

    try {
      let data;

      if (searchId) {
        data = await searchPhotosById(searchId, offset);
      } else if (image) {
        data = await searchPhotosByImage(image, offset, size, dim);
      } else {
        data = await searchPhotos(q, offset, dim);
      }

      if (id !== requestId.current) return;

      setTotal(data.total);
      setSearchId(data.search_id || searchId || null);
      setResults((prev) =>
        offset === 0 ? data.results : [...prev, ...data.results],
      );
    } catch (e) {
      if (id === requestId.current) setError(e.message);
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }, []);

  const search = useCallback(
    (text, dim, size) => {
      const q = text.trim();
      if (!q) return;
      setQuery(q);
      setResults([]);
      setTotal(0);
      window.scrollTo(0, 0);
      run(q, 0, dim, null, null, size);
    },
    [run],
  );

  const searchImage = useCallback(
    (file, dim, size) => {
      if (!file) return;

      imageRef.current = file;
      setQuery("");
      setResults([]);
      setTotal(0);
      window.scrollTo(0, 0);
      run("", 0, dim, file, null, size);
    },
    [run],
  );

  const loadMore = useCallback((dim, size) => {
    if (loading || results.length >= total) return;
    // run(query, results.length, imageRef.current);
    console.log("searchId in loadMore:", searchId);
    run(query, results.length, dim, null, searchId, size);
  }, [loading, results.length, total, query, searchId, run]);

  return {
    query,
    results,
    total,
    loading,
    error,
    searchId,
    search,
    loadMore,
    searchImage,
  };
}
