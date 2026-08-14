import cv2
import numpy
import onnxruntime


class YOLOModel:
    # Initialization
    def __init__(self, detectionThreshold: float, model_path: str):
        # super().__init__()
        self._textForegroundColor = (0, 0, 0)
        self._textBackgroundColor = (255, 255, 255)
        self._fillColor = (255, 0, 0)

        self._session = onnxruntime.InferenceSession(model_path, providers=self.GetProviders())
        self._inputWidth = self._session.get_inputs()[0].shape[3]
        self._inputHeight = self._session.get_inputs()[0].shape[2]
        self._detectionThreshold = detectionThreshold

        self._outputWidth = self._session.get_outputs()[1].shape[3]
        self._outputHeight = self._session.get_outputs()[1].shape[2]
        print(self._outputWidth)
        # self._outputHeight = self._session.get_outputs()[0].shape[2]

        print(f"YOLOModel defined as: {vars(self)}")

    # Load classes

    # Return ONNX Runtime providers
    def GetProviders(self) -> list[str]:
        print(f"Available ONNX Runtime providers: {onnxruntime.get_available_providers()}")
        provider = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider", "DmlExecutionProvider") if provider in onnxruntime.get_available_providers()]
        print(f"Using ONNX Runtime providers: {provider}")
        return provider

    # Execute model and return the drawn overlay (keeps old behavior for callers that want an annotated image)
    def Execute(self, image: numpy.ndarray) -> numpy.ndarray:
        detections = self.Predict(image)
        if detections:
            return self.DrawDetections(image, detections)
        return image


    # Return model input
    def GetInput(self, image: numpy.ndarray) -> tuple[numpy.ndarray, tuple[int, int]]:
        (image, padding) = self.LetterBoxInputImage(image, self._inputWidth, self._inputHeight)
        image = numpy.array(image) / 255.0
        image = numpy.transpose(image, (2, 0, 1))
        image = numpy.expand_dims(image, axis=0)
        image = image.astype(numpy.float32)
        return (image, padding)

    # Resize and reshape image while maintaining aspect ratio by adding padding
    def LetterBoxInputImage(self, image: numpy.ndarray, targetWidth: int, targetHeight: int) -> tuple[numpy.ndarray, tuple[int, int]]:
        shape = image.shape[:2]
        (imageHeight, imageWidth) = shape
        scaleRatio = min(targetHeight / imageHeight, targetWidth / imageWidth)
        padding = round(imageWidth * scaleRatio), round(imageHeight * scaleRatio)
        (paddingWidth, paddingHeight) = padding
        if shape[::-1] != padding:
            image = cv2.resize(image, padding, interpolation=cv2.INTER_LINEAR)
        (deltaWidth, deltaHeight) = (targetWidth - paddingWidth) / 2, (targetHeight - paddingHeight) / 2
        (top, bottom) = round(deltaHeight - 0.1), round(deltaHeight + 0.1)
        (left, right) = round(deltaWidth - 0.1), round(deltaWidth + 0.1)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return image, (top, left)

    # Process model output (YOLOv10+) into a drawn overlay + segment canvas
    def ProcessOutput(self, image: numpy.ndarray, output, padding: tuple[int, int]) -> numpy.ndarray:
        detections = self.ProcessDetections(image, output, padding)
        if detections:
            return self.DrawDetections(image, detections)
        return image

    # Process model output (YOLOv10+)
    # Turn raw model output (YOLOv10+) into a list of per-instance detections, without drawing.
    def ProcessDetections(self, image: numpy.ndarray, output, padding: tuple[int, int]) -> list[dict]:
        predictions = output[0]
        predictions = predictions[0]
        boundingBoxes = predictions[:, :4]
        classScores = predictions[:, 4]
        classIndices = predictions[:, 5]
        maskCoefficients = predictions[:, 6:]
        indices = numpy.where(classScores >= self._detectionThreshold)[0]

        if len(indices) == 0:
            return []

        prototypeMasks = output[1]
        prototypeMasks = prototypeMasks[0]
        prototypeMasks = prototypeMasks.reshape(32, -1)

        filteredMaskCoefficients = maskCoefficients[indices]
        filteredClassIndices = classIndices[indices]
        filteredClassScores = classScores[indices]
        filteredBoundingBoxes = boundingBoxes[indices]

        (imageHeight, imageWidth) = image.shape[:2]
        scale = min(self._inputHeight / imageHeight, self._inputWidth / imageWidth)
        filteredBoundingBoxes[:, 0] = (filteredBoundingBoxes[:, 0] - padding[1]) / scale
        filteredBoundingBoxes[:, 1] = (filteredBoundingBoxes[:, 1] - padding[0]) / scale
        filteredBoundingBoxes[:, 2] = (filteredBoundingBoxes[:, 2] - padding[1]) / scale
        filteredBoundingBoxes[:, 3] = (filteredBoundingBoxes[:, 3] - padding[0]) / scale

        masks = self.ProcessMasks(
            filteredMaskCoefficients @ prototypeMasks,
            filteredBoundingBoxes,
            imageWidth, imageHeight, padding, scale,
        )

        return [
            {
                "box": filteredBoundingBoxes[i].astype(int).tolist(),
                "class": int(filteredClassIndices[i]),
                "score": float(filteredClassScores[i]),
                "mask": masks[i],
            }
            for i in range(len(filteredBoundingBoxes))
        ]



    # Run inference and return structured detections (box, class, score, mask) without drawing anything.
    # "mask" is a boolean HxW array the same size as the input image.
    def Predict(self, image: numpy.ndarray) -> list[dict]:
        (input, padding) = self.GetInput(image)
        print(f"Input shape: {input.shape}, padding: {padding}")
        output = self._session.run(None, {self._session.get_inputs()[0].name: input})
        return self.ProcessDetections(image, output, padding)


    # Draw detections
    def DrawDetections(self, image: numpy.ndarray, indices, detections, masks, boundingBoxes, padding: tuple[int, int], scale: float) -> numpy.ndarray:
        (imageHeight, imageWidth) = image.shape[:2]
        segments = self.ProcessMasks(masks, boundingBoxes, imageWidth, imageHeight, padding, scale)
        segmentCanvas = image.copy()
        for index, segment in zip(indices, segments):
            (x1, y1, x2, y2) = detections[index]["box"]
            self.DrawBoundingBox(image, f"{detections[index]["class"]}: {detections[index]["score"]:.0%}", x1, y1, x2, y2)
            segmentCanvas[segment] = self._fillColor
        return cv2.addWeighted(segmentCanvas, 0.3, image, 0.7, 0)

    # Draw bounding box
    def DrawBoundingBox(self, image: numpy.ndarray, title: str, x1: int, y1: int, x2: int, y2: int):
        fontScale = 0.5
        thickness = 1
        (textWidth, textHeight), baseline = cv2.getTextSize(title, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, thickness=thickness)
        if y1 - textHeight >= 0:
            yTitle1 = y1 - textHeight - 2 * baseline
            yTitle2 = y1 - baseline
        else:
            yTitle1 = y1 + textHeight + 2 * baseline
            yTitle2 = y1 + textHeight + baseline
        cv2.rectangle(image, (x1, yTitle1), (x1 + textWidth, y1), self._textBackgroundColor, -1)
        cv2.putText(image, title, (x1, yTitle2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=self._textForegroundColor, thickness=thickness)
        cv2.rectangle(image, (x1, y1), (x2, y2), self._textBackgroundColor, thickness=thickness)

    # Process masks
    def ProcessMasks(self, masks, boundingBoxes, imageWidth: int, imageHeight: int, padding: tuple[int, int], scale: float):
        masks = masks.reshape(-1, self._outputWidth, self._outputHeight)
        masks = masks.transpose(1, 2, 0)
        masks = self.ScaleMasks(masks, imageWidth, imageHeight, padding, scale)
        masks = numpy.einsum("HWN -> NHW", masks)
        masks = masks > 0.0
        masks = self.CropMasks(masks, boundingBoxes)
        return masks

    # Scale masks to the original image size
    def ScaleMasks(self, masks, imageWidth: int, imageHeight: int, padding: tuple[int, int], scale: float):
        length = max((imageWidth, imageHeight))
        masks = cv2.resize(masks, (length, length), interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = numpy.expand_dims(masks, axis=-1)
        if padding[0] > 0:
            masks = self.ShiftMasks(masks, int(-padding[0] / scale), 0)
        if padding[1] > 0:
            masks = self.ShiftMasks(masks, int(-padding[1] / scale), 1)
        masks = masks[:imageHeight, :imageWidth]
        return masks

    # # Crop masks to the bounding boxes
    def CropMasks(self, masks, boundingBoxes):
        numberOfDetections, masksHeight, masksWidth = masks.shape
        cropped = numpy.zeros((numberOfDetections, masksHeight, masksWidth), dtype=numpy.bool_)
        for i, (boundingBoxLeft, boundingBoxTop, boundingBoxRight, boundingBoxBottom) in enumerate(boundingBoxes):
            boundingBoxLeft, boundingBoxTop, boundingBoxRight, boundingBoxBottom = map(int, (boundingBoxLeft, boundingBoxTop, boundingBoxRight, boundingBoxBottom))
            boundingBoxLeft = max(boundingBoxLeft, 0)
            boundingBoxRight = min(boundingBoxRight, masksWidth - 1)
            boundingBoxTop = max(boundingBoxTop, 0)
            boundingBoxBottom = min(boundingBoxBottom, masksHeight - 1)
            cropped[i, boundingBoxTop:boundingBoxBottom, boundingBoxLeft:boundingBoxRight] = masks[i, boundingBoxTop:boundingBoxBottom, boundingBoxLeft:boundingBoxRight]
        return cropped

    # Shift masks
    def ShiftMasks(self, masks, shiftAmount: int, shiftAxis: int, fillValue=0):
        shiftedMasks = numpy.full_like(masks, fillValue)
        if shiftAxis == 0:
            if shiftAmount > 0:
                shiftedMasks[shiftAmount:, :, :] = masks[:-shiftAmount, :, :]
            elif shiftAmount < 0:
                shiftedMasks[:shiftAmount, :, :] = masks[-shiftAmount:, :, :]
            else:
                shiftedMasks = masks
        elif shiftAxis == 1:
            if shiftAmount > 0:
                shiftedMasks[:, shiftAmount:, :] = masks[:, :-shiftAmount, :]
            elif shiftAmount < 0:
                shiftedMasks[:, :shiftAmount, :] = masks[:, -shiftAmount:, :]
            else:
                shiftedMasks = masks
        elif shiftAxis == 2:
            if shiftAmount > 0:
                shiftedMasks[:, :, shiftAmount:] = masks[:, :, :-shiftAmount]
            elif shiftAmount < 0:
                shiftedMasks[:, :, :shiftAmount] = masks[:, :, -shiftAmount:]
            else:
                shiftedMasks = masks
        else:
            raise ValueError("Invalid shift axis specified.")
        return shiftedMasks

