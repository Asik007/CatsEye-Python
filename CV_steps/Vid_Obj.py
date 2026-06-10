
import os

import av
import numpy as np
from fractions import Fraction


class VidObj:
    """
    A video object built on PyAV that is intercompatible with OpenCV.

    Frames are decoded and returned as uint8 numpy arrays in BGR channel order,
    matching OpenCV's native format so they can be passed directly to any
    cv2 function without conversion.
    """

    FPS: float = 0.0
    duration: float = 0.0  # seconds
    len: int = 0            # total frames
    codec: str = ""
    res: list = [0, 0]      # [width, height]
    # colorspace: str = ""
    # HDR: bool = False
    bitdepth: int = 8
    use_increased_depth: bool = False  # Whether to attempt to preserve bit depth > 8 bits per channel

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __new__(cls, *args, **kwargs):
        """Properly allocate the instance (original stub returned None)."""
        instance = super().__new__(cls)
        # print("Creating a new VidObj")
        return instance

    def __init__(self, vid_path: str):
        self.path = vid_path
        # self.container = av.open(vid_path)
        # self.container.streams.video[0].thread_type = "AUTO"  # Go faster!
        # self.video_stream = self.container.streams.video[0]

        info = self._get_info(vid_path)
        for key, value in info.items():
            setattr(self, key, value)

    # ------------------------------------------------------------------ #
    #  Info extraction                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_info(media_path: str) -> dict:
        """
        Open the file, pull metadata from the first video stream, and return
        a dict whose keys match the class attributes.
        """

        vid_info = {}

        with av.open(media_path) as container:
            stream = container.streams.video[0]

            # --- FPS ---------------------------------------------------
            fps_frac: Fraction = stream.average_rate
            vid_info["FPS"] = float(fps_frac) if fps_frac else 0.0

            # --- Duration (seconds) ------------------------------------
            if stream.duration and stream.time_base:
                vid_info["duration"] = float(stream.duration * stream.time_base)
            elif container.duration:
                vid_info["duration"] = container.duration / av.time_base
            else:
                vid_info["duration"] = 0.0

            # --- Frame count -------------------------------------------
            if stream.frames:
                vid_info["len"] = int(stream.frames)
            elif vid_info["FPS"] and vid_info["duration"]:
                vid_info["len"] = int(vid_info["FPS"] * vid_info["duration"])
            else:
                vid_info["len"] = 0

            # --- Codec name --------------------------------------------
            vid_info["codec"] = stream.codec_context.name  # e.g. "h264"

            # --- Resolution [width, height] ----------------------------
            vid_info["res"] = [stream.codec_context.width,
                               stream.codec_context.height]

            # # --- Color space -------------------------------------------
            # # color_space is a string like "bt709"; may be None for some files
            # vid_info["color_space"] = (
            #     stream.codec_context.colorspace or "unknown"
            # )

            # --- Format ------------------------------------------------
            vid_info["format"] = stream.codec_context.format.name if stream.codec_context.format else "unknown"

            # --- HDR flag ------------------------------------------------
            # A simple heuristic: if the color space is a known HDR type, set HDR=True
            # hdr_color_spaces = {"bt2020nc", "bt2020c", "bt2100"}
            # vid_info["HDR"] = stream.colorspace in hdr_color_spaces

            # --- Bit depth ------------------------------------------------
            # Common formats: "yuv420p" (8-bit), "yuv420p10le" (10-bit), etc.
            vid_info["bitdepth"] = stream.codec_context.format.components[0].bits if stream.codec_context.format else 8
        return vid_info

    # ------------------------------------------------------------------ #
    #  Frame iteration  (yields OpenCV-compatible BGR numpy arrays)        #
    # ------------------------------------------------------------------ #

    def frames_gen(self, start: int = 0, end: int = None):
        """
        Generator that yields decoded frames as uint8 BGR numpy arrays.

        Parameters
        ----------
        start : int
            First frame index to yield (0-based).
        end : int | None
            Last frame index (exclusive). None means read to end of file.

        Yields
        ------
        np.ndarray
            Shape (height, width, 3), dtype uint8, channel order BGR.
        """
        end = end if end is not None else self.len
        container = av.open(self.path)
        stream = container.streams.video[0]

        # Enable fast seek / threaded decode for performance
        stream.thread_type = "AUTO"

        frame_idx = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_idx < start:
                    frame_idx += 1
                    print(f" starting frame gen {frame_idx}")
                    continue
                if frame_idx >= end:
                    container.close()
                    print(f" end of frames: {frame_idx-1}")
                    return
                
                print(f" yielding frame {frame_idx}")
                yield self._frame_to_bgr(frame)
                frame_idx += 1

        container.close()

    def read_frame(self, frame_number: int) -> np.ndarray | None:
        """
        Seek to and decode a single frame by index.

        Returns
        -------
        np.ndarray | None
            BGR numpy array, or None if frame_number is out of range.
        """
        if frame_number < 0 or frame_number >= self.len:
            return None

        container = av.open(self.path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        # Convert frame index → timestamp in stream time_base units
        if self.FPS and stream.time_base:
            pts = int(frame_number / self.FPS / float(stream.time_base))
            container.seek(pts, stream=stream, any_frame=False, backward=True)

        result = None
        current = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if current == frame_number or result is None:
                    result = self._frame_to_bgr(frame)
                    if current >= frame_number:
                        container.close()
                        return result
                current += 1

        container.close()
        return result

    def get_mp4(self, output_path: str) -> None:
        """
        Re-encode the video to MP4 format at the given path.

        This can be useful for compatibility with libraries that require MP4 input.
        """
        input_container = av.open(self.path)
        output_container = av.open(output_path, mode="w")

        input_stream = input_container.streams.video[0]
        output_stream = output_container.add_stream("h264", rate=input_stream.rate)
        output_stream.width = input_stream.width
        output_stream.height = input_stream.height
        # output_stream.pix_fmt = "yuv420p"

        for packet in input_container.demux(input_stream):
            for frame in packet.decode():
                frame.pts = None  # Let encoder set PTS
                for packet in output_stream.encode(frame):
                    output_container.mux(packet)

        # Flush encoder
        for packet in output_stream.encode():
            output_container.mux(packet)

        input_container.close()
        output_container.close()
    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _frame_to_bgr(self, frame: av.VideoFrame) -> np.ndarray:
        """Convert an av.VideoFrame to a uint8 BGR numpy array."""
        # reformat() uses libswscale — fast, handles any pixel format
        bgr_frame = frame.reformat(format="bgr24")
        if self.use_increased_depth:
            # TODO: WTF is the correct way to do this?
            print("Attempting to preserve increased bit depth (not fully implemented) |||| will return the same result as normal for now")
        return bgr_frame.to_ndarray()  # shape: (H, W, 3)

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                      #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.len

    def __repr__(self) -> str:
        return (
            f"VidObj(codec={self.codec!r}, res={self.res}, "
            f"FPS={self.FPS:.2f}, duration={self.duration:.2f}s, "
            f"frames={self.len})"
        )

    def __del__(self):
        try:
            self._container.close()
        except Exception:
            pass


if __name__ == "__main__":

    for file in os.listdir("uploads"):
        if file.endswith((".mov", ".MOV")):
            print(f"Testing VidObj with {file}")
            vid = VidObj(os.path.join("uploads", file))
            print(vars(vid))
            # for frame in vid.frames_gen():
            #     print(frame.shape)
            #     break  # Just test the first frame
    # print(vid)

    # index = 0
    # for frame in vid.frames_gen():
    #     index += 1
    #     # print(frame.shape)