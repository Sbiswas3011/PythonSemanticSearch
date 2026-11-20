import { useState, useRef } from "react";
import { RiGalleryFill } from "react-icons/ri";
import { FaCamera } from "react-icons/fa";

export default function SearchBar({ onSearch, onOpenCamera, dim, setDim, onImageSearch }) {
  const [text, setText] = useState("");
  const galleryRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(text, dim);
  };

  const handleGallerySelect = (e) => {
    const file = e.target.files?.[0];

    if (!file) return;

    onImageSearch(file, dim);

    // Allows selecting the same image again.
    e.target.value = "";
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
          <FaCamera />
        </button>

        <input
          ref={galleryRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          hidden
          onChange={handleGallerySelect}
        />

        <button className="gallery-button" type="button" onClick={() => galleryRef.current?.click()}>
          <RiGalleryFill />
        </button>
      </form>
    </header>
  );
}
