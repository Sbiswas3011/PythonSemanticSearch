import { useEffect } from "react";
import { imgUrl } from "../api";
import CorrectionForm from "./CorrectionForm.jsx";

export default function Lightbox({ photo, query, onClose }) {
  useEffect(() => {
    if (!photo) return;
    const onKey = e => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [photo, onClose]);

  if (!photo) return null;

  return (
    <div id="lightbox" onClick={onClose}>
      <div className="lightbox-content" onClick={e => e.stopPropagation()}>
        <img src={imgUrl(photo.id)} alt={photo.id} onClick={onClose} />
        <div className="cap">{photo.id}   {photo.raw_path}</div>
        {/* key resets the form whenever a different photo is opened */}
        <CorrectionForm key={photo.id} photo={photo} query={query} />
      </div>
    </div>
  );
}
