from collections import deque
import cv2
import numpy as np

class Batching:
    """
    Collect 5 frames, sum them, and remap the result back to 0-255.
    Returns None until 5 frames have been collected.
    """

    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.frames = deque(maxlen=batch_size)

    def reset(self) -> None:
        self.frames.clear()

    def process(self, frame):
        self.frames.append(frame)

        if len(self.frames) < self.batch_size:
            return None

        stacked = np.stack(self.frames, axis=0).astype(np.float32)
        summed = np.sum(stacked, axis=0)

        remapped = cv2.normalize(summed, None, 0, 255, cv2.NORM_MINMAX)
        return remapped.astype(np.uint8)

    def __call__(self, frame):
        def create_batching(batch_size: int = 5):
            frames = deque(maxlen=batch_size)

            def reset() -> None:
                frames.clear()

            def process(frame):
                frames.append(frame)

                if len(frames) < batch_size:
                    return None

                stacked = np.stack(frames, axis=0).astype(np.float32)
                summed = np.sum(stacked, axis=0)
                remapped = cv2.normalize(summed, None, 0, 255, cv2.NORM_MINMAX)
                return remapped.astype(np.uint8)

            return {
                "reset": reset,
                "process": process,
            }