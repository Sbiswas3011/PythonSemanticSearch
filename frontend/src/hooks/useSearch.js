import { useState, useCallback, useRef } from "react";
import { searchPhotos } from "../api";

export function useSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const requestId = useRef(0); // ignore responses from outdated searches

  const run = useCallback(async (q, offset) => {
    const id = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const data = await searchPhotos(q, offset);
      if (id !== requestId.current) return;
      setTotal(data.total);
      setResults(prev => (offset === 0 ? data.results : [...prev, ...data.results]));
    } catch (e) {
      if (id === requestId.current) setError(e.message);
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }, []);

  const search = useCallback(
    text => {
      const q = text.trim();
      if (!q) return;
      setQuery(q);
      setResults([]);
      setTotal(0);
      window.scrollTo(0, 0);
      run(q, 0);
    },
    [run]
  );

  const loadMore = useCallback(() => {
    if (loading || results.length >= total) return;
    run(query, results.length);
  }, [loading, results.length, total, query, run]);

  return { query, results, total, loading, error, search, loadMore };
}
