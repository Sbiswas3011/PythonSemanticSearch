import { useRef, useEffect, useState } from "react";

export default function Camera({
  onImageSearch,
  onClose,
  onError,
  dim,
  quality,
  setQuality,
}) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const activeRef = useRef(true);
  const requestRef = useRef(0);
  const nativeCameraRef = useRef(null);
  // const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  const isMobile = false; // Disable mobile camera for testing

  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState("");
  const [preview, setPreview] = useState(null);

  const qualitySettings = {
    tiny: 0.1,
    small: 0.3,
    medium: 0.6,
    large: 0.9,
    original: 1.0,
  };

  // Stop the currently active camera
  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  // Discover available cameras
  const loadCameras = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();

      if (!activeRef.current) return;

      const videoDevices = devices
        .filter((device) => device.kind === "videoinput")
        .map((device, index) => ({
          deviceId: device.deviceId,
          label: device.label || `Camera ${index + 1}`,
        }));

      console.log(videoDevices);
      setCameras(videoDevices);
    } catch (err) {
      console.error("Unable to enumerate cameras:", err);
    }
  };

  // Start a particular camera
  const startCamera = async (deviceId = null) => {
    const requestId = ++requestRef.current;

    try {
      stopCamera();

      const videoConstraints = deviceId
        ? { deviceId: { exact: deviceId } }
        : { facingMode: { ideal: "environment" } };

      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });

      // Ignore outdated requests or a closed camera component
      if (!activeRef.current || requestId !== requestRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Get the actual device ID selected by the browser
      const actualDeviceId = stream.getVideoTracks()[0].getSettings().deviceId;

      if (actualDeviceId) {
        setSelectedCamera(actualDeviceId);
      }

      // Refresh the list after permission is granted
      await loadCameras();
    } catch (err) {
      if (!activeRef.current || requestId !== requestRef.current) return;

      console.error("Camera initialization failed:", err);
      onError(err.message);
    }
  };

  // Initial camera initialization
  useEffect(() => {
    activeRef.current = true;

    if (isMobile) {
      nativeCameraRef.current?.click();
      return () => {
        activeRef.current = false;
      };
    }

    const initialize = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("Camera API unavailable. Use HTTPS or localhost.");
        }

        await startCamera();
      } catch (err) {
        if (!activeRef.current) return;

        console.error("Camera initialization failed:", err);
        onClose();
        onError(err.message);
      }
    };

    initialize();

    // Update camera list when devices are connected/disconnected
    navigator.mediaDevices?.addEventListener("devicechange", loadCameras);

    return () => {
      activeRef.current = false;
      requestRef.current++;

      navigator.mediaDevices?.removeEventListener("devicechange", loadCameras);

      stopCamera();
    };
  }, [onClose, onError]);

  // Switch to another camera
  const handleCameraChange = (event) => {
    const deviceId = event.target.value;

    if (deviceId) {
      startCamera(deviceId);
    }
  };

  const captureImage = () => {
    const video = videoRef.current;

    if (!video || !video.videoWidth) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    // canvas.toBlob(
    //   (blob) => {
    //     if (blob) {
    //       const file = new File([blob], "capture.jpg", {
    //         type: "image/jpeg",
    //       });

    //       onImageSearch(file, dim);
    //       onClose();
    //     }
    //   },
    //   "image/jpeg",
    //   qualitySettings[quality],
    // );

    canvas.toBlob(
      (blob) => {
        if (!blob) return;

        const file = new File([blob], "capture.jpg", {
          type: "image/jpeg",
        });

        setPreview({
          file,
          url: URL.createObjectURL(blob),
        });
      },
      "image/jpeg",
      qualitySettings[quality],
    );
  };

  const retakeImage = () => {
    URL.revokeObjectURL(preview.url);
    setPreview(null);
  };

  const submitImage = () => {
    onImageSearch(preview.file, dim);
    URL.revokeObjectURL(preview.url);
    onClose();
  };

  useEffect(() => {
    if (!preview && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play();
    }
  }, [preview]);

  const handleNativeCapture = (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    setPreview({
      file,
      url: URL.createObjectURL(file),
    });

    event.target.value = "";
  };

  return (
    <div className="camera card">
      <input
        ref={nativeCameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleNativeCapture}
        hidden
      />

      {preview ? (
        <>
          <img
            src={preview.url}
            alt="Captured preview"
            className="camera-preview"
          />

          <div className="camera-actions">
            <button type="button" onClick={submitImage}>
              Use Photo
            </button>

            <button
              type="button"
              onClick={() => {
                if (isMobile) {
                  nativeCameraRef.current?.click();
                } else {
                  retakeImage();
                }
              }}
            >
              Retake
            </button>
          </div>
        </>
      ) : !isMobile ? (
        <>
          <video ref={videoRef} autoPlay playsInline muted />

          <div className="camera-controls">
            {cameras.length > 1 && (
              <div className="camera-selection">
                <label htmlFor="camera-select">Camera</label>

                <select
                  id="camera-select"
                  value={selectedCamera}
                  onChange={handleCameraChange}
                >
                  {cameras.map((camera) => (
                    <option key={camera.deviceId} value={camera.deviceId}>
                      {camera.label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="quality-control">
              <div className="quality-options">
                {["tiny", "small", "medium", "large", "original"].map(
                  (option) => (
                    <button
                      key={option}
                      type="button"
                      className={quality === option ? "active" : ""}
                      onClick={() => setQuality(option)}
                    >
                      {option.charAt(0).toUpperCase() + option.slice(1)}
                    </button>
                  ),
                )}
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
        </>
      ) : null}
    </div>
  );
}
