"""
Base class for all segmentation model wrappers.
Ensures unified interface across different model outputs.
"""


class BaseSegmentationModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        """
        Predict method to be implemented by all subclasses.
        Should return a segmentation mask.
        """
        raise NotImplementedError("Subclasses must implement this method.")
