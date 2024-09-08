# detection.py

import torch
import torchvision
import torchvision.transforms as T

class FasterRCNNDetector:
    def __init__(self):
        # load the Faster R-CNN model from torchvision
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()  # Set the model to evaluation mode

    def detect(self, image):
        """Detect objects in an image."""
        transform = T.Compose([
            T.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        return self._filter_predictions(predictions[0])

    def _filter_predictions(self, prediction):
        """Filter the predictions based on the detection threshold."""
        boxes = prediction['boxes'].numpy()
        scores = prediction['scores'].numpy()
        labels = prediction['labels'].numpy()

        mask = scores >= 0.5  # Threshold is set to 0.5 by default
        return boxes[mask], scores[mask], labels[mask]
