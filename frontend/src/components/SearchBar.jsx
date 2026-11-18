import { useState } from "react";

export default function SearchBar({ onSearch, onOpenCamera, dim, setDim, quality, setQuality }) {
  const [text, setText] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    setQuality("original"); // Reset quality to "original" on new search
    onSearch(text, dim, quality);
  };

  return (
    <header>
      <form className="searchbar" onSubmit={handleSubmit}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Describe a photo, e.g. a flamingo"
          autoFocus
        />

        <label className="dimension-select">
          <select value={dim} onChange={(e) => setDim(Number(e.target.value))}>
            <option value={0}>768D</option>
            <option value={1}>512D</option>
            <option value={2}>256D</option>
            <option value={3}>128D</option>
          </select>
        </label>

        <button type="submit">Search</button>

        <button className="camera-button" type="button" onClick={onOpenCamera}>
          📷
        </button>
      </form>
    </header>
  );
}
