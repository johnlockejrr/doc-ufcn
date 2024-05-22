from dataclasses import dataclass, field

from doc_ufcn import models
from doc_ufcn.main import DocUFCN


@dataclass
class UFCNModel:
    name: str
    colors: list
    title: str
    description: str
    classes: list = field(default_factory=list)
    model: DocUFCN = None

    def get_class_name(self, channel_idx):
        return self.classes[channel_idx]

    @property
    def loaded(self):
        return self.model is not None

    @property
    def num_channels(self):
        return len(self.classes)

    def load(self):
        # Download the model
        model_path, parameters = models.download_model(name=self.name)

        # Store classes
        self.classes = parameters["classes"]

        # Check that the number of colors is equal to the number of classes -1
        assert (
            self.num_channels - 1 == len(self.colors)
        ), f"The parameter classes_colors was filled with the wrong number of colors. {self.num_channels-1} colors are expected instead of {len(self.colors)}."

        # Load the model
        self.model = DocUFCN(
            no_of_classes=len(self.classes),
            model_input_size=parameters["input_size"],
            device="cpu",
        )
        self.model.load(
            model_path=model_path, mean=parameters["mean"], std=parameters["std"]
        )
