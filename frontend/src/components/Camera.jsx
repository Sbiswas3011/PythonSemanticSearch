import { useRef, useEffect, useState } from "react";

export default function Camera({ onImageSearch, onClose, onError, dim, quality, setQuality }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const activeRef = useRef(true);

  // const [quality, setQuality] = useState("tiny");

  const qualitySettings = {
    tiny: 0.1,
    small: 0.3,
    medium: 0.6,
    large: 0.9,
    original: 1.0,
  };

  useEffect(() => {
    activeRef.current = true;

    const startCamera = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("Camera API unavailable. Use HTTPS or localhost.");
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false,
        });

        if (!activeRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        if (!activeRef.current) return;

        console.error("Camera initialization failed:", err);
        onClose();
        onError(err.message);
      }
    };

    startCamera();

    return () => {
      activeRef.current = false;

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [onClose]);

  const captureImage = () => {
    const video = videoRef.current;

    if (!video || !video.videoWidth) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (blob) {
          const file = new File([blob], "capture.jpg", {
            type: "image/jpeg",
          });

          onImageSearch(file, dim, quality);
          onClose();
        }
      },
      "image/jpeg",
      qualitySettings[quality],
    );
  };

  return (
    <div className="camera card">
      <video ref={videoRef} autoPlay playsInline muted />

      <div className="camera-controls">
        <div className="quality-control">
          <div className="quality-options">
            {["tiny", "small", "medium", "large", "original"].map((option) => (
              <button
                key={option}
                type="button"
                className={quality === option ? "active" : ""}
                onClick={() => setQuality(option)}
              >
                {option.charAt(0).toUpperCase() + option.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div className="camera-actions">
          <button type="button" onClick={captureImage}>
            Capture
          </button>

          <button type="button" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
