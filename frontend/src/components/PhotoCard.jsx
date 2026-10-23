import { imgUrl } from "../api";

const line = parts => parts.filter(Boolean).join("  ·  ");

export default function PhotoCard({ photo, onSelect }) {
  const r = photo;
  return (
    <div className="card" title={r.raw_path} onClick={() => onSelect(r)}>
      <img src={imgUrl(r.id)} loading="lazy" alt={r.id} />
      <div className="info">
        <b>{r.id} ({r.score.toFixed(3)})</b>
        <br />
        {line([
          r.date ? r.date.replace("T", " ") : "",
          r.iso ? "ISO " + r.iso : "",
          r.focal ? r.focal + "mm" : "",
          r.f ? "f/" + r.f : "",
          r.exposure || ""
        ])}
      </div>
    </div>
  );
}
