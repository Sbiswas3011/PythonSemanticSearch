import { useRef, useEffect } from "react";

export default function Camera({ onImageSearch, onClose }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    let active = true;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false
        });

        if (!active) {
          stream.getTracks().forEach(track => track.stop());
          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (err) {
        onClose();
        alert("Unable to access camera: " + err.message);
      }
    };

    startCamera();

    return () => {
      active = false;
      streamRef.current?.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    };
  }, []);

  const captureImage = () => {
    const video = videoRef.current;

    if (!video || !video.videoWidth) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext("2d").drawImage(
      video, 0, 0, canvas.width, canvas.height
    );

    canvas.toBlob(blob => {
      if (blob) {
        const file = new File([blob], "capture.jpg", {
          type: "image/jpeg"
        });

        onImageSearch(file);
        onClose();
      }
    }, "image/jpeg", 0.9);
  };

  return (
    <div className="camera card">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
      />

      <div className="info">
        <button type="button" onClick={captureImage}>
          Capture
        </button>

        <button type="button" onClick={onClose}>
          Cancel
        </button>
      </div>
    </div>
  );
}


// import { useRef, useEffect } from "react";

// export default function Camera({ onImageSearch, onClose }) {
//   const videoRef = useRef(null);
//   const streamRef = useRef(null);

//   useEffect(() => {
//     let active = true;

//     async function startCamera() {
//       try {
//         console.log("Starting camera");

//         const stream = await navigator.mediaDevices.getUserMedia({
//           video: true,
//           audio: false
//         });

//         if (!active) {
//           stream.getTracks().forEach(track => track.stop());
//           return;
//         }

//         streamRef.current = stream;
//         videoRef.current.srcObject = stream;

//         await videoRef.current.play();

//         console.log("Video playing");
//       } catch (err) {
//         console.error("Camera failed:", err);
//       }
//     }

//     startCamera();

//     return () => {
//       active = false;
//       streamRef.current?.getTracks().forEach(track => track.stop());
//       streamRef.current = null;
//     };
//   }, []);

//   return (
//     <div className="camera card">
//       <video ref={videoRef} autoPlay playsInline muted />

//       <div className="info">
//         <button type="button" onClick={() => onImageSearch(null)}>
//           Capture
//         </button>

//         <button type="button" onClick={onClose}>
//           Cancel
//         </button>
//       </div>
//     </div>
//   );
// }
