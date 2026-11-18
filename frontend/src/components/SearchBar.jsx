import { useState } from "react";

export default function SearchBar({ onSearch, onOpenCamera }) {
  const [text, setText] = useState("");

  const handleSubmit = e => {
    e.preventDefault();
    onSearch(text);
  };

  return (
    <header>
      <form className="searchbar" onSubmit={handleSubmit}>
        <input
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Describe a photo, e.g. a flamingo"
          autoFocus
        />

        <button type="submit">Search</button>

        <button className="camera-button" type="button" onClick={onOpenCamera}>
          📷
        </button>
      </form>
    </header>
  );
}
