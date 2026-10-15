"""Show the live USB webcam feed. Press q or Esc to quit."""
import cv2

DEVICE = 0


def main() -> None:
    cap = cv2.VideoCapture(DEVICE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{DEVICE}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed")
                break
            cv2.imshow("webcam", frame[frame.shape[0] // 2 :])
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
