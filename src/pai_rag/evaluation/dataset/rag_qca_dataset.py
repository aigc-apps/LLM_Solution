from typing import List, Optional
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llama_dataset.base import BaseLlamaDataExample
from llama_index.core.llama_dataset import CreatedBy
import json
from llama_index.core.bridge.pydantic import BaseModel
from loguru import logger


class RagQcaSample(BaseLlamaDataExample):
    """Predicted RAG example class. Analogous to traditional ML datasets, this dataset contains
    the "features" (i.e., query + context) to make a prediction and the "label" (i.e., response)
    to evaluate the prediction.
    """

    query: str = Field(default=str, description="The user query for the example.")
    query_by: Optional[CreatedBy] = Field(
        default=None, description="What generated the query."
    )
    reference_contexts: Optional[List[str]] = Field(
        default=None,
        description="The contexts used to generate the reference answer.",
    )
    reference_node_ids: Optional[List[str]] = Field(
        default=None, description="The node id corresponding to the contexts"
    )
    reference_image_url_list: Optional[List[str]] = Field(
        default=None,
        description="The image urls used to generate the reference answer.",
    )
    reference_answer: str = Field(
        default=str,
        description="The reference (ground-truth) answer to the example.",
    )
    reference_answer_by: Optional[CreatedBy] = Field(
        default=None, description="What model generated the reference answer."
    )

    predicted_contexts: Optional[List[str]] = Field(
        default=None,
        description="The contexts used to generate the predicted answer.",
    )
    predicted_node_ids: Optional[List[str]] = Field(
        default=None,
        description="The node id corresponding to the predicted contexts",
    )
    predicted_node_scores: Optional[List[float]] = Field(
        default=None,
        description="The node score corresponding to the predicted contexts",
    )
    predicted_image_url_list: Optional[List[str]] = Field(
        default=None,
        description="The image urls used to generate the reference answer.",
    )
    predicted_answer: str = Field(
        default="",
        description="The predicted answer to the example.",
    )
    predicted_answer_by: Optional[CreatedBy] = Field(
        default=None, description="What model generated the predicted answer."
    )

    @property
    def class_name(self) -> str:
        """Data example class name."""
        return "RagQcaSample"


class PaiRagQcaDataset(BaseModel):
    examples: List[RagQcaSample] = Field(
        default_factory=list, description="Data examples of this dataset."
    )
    labelled: bool = Field(
        default=False, description="Whether the dataset is labelled or not."
    )
    predicted: bool = Field(
        default=False, description="Whether the dataset is predicted or not."
    )

    @property
    def class_name(self) -> str:
        """Class name."""
        return "PaiRagQcaDataset"

    def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w", encoding="utf-8") as f:
            examples = [el.model_dump() for el in self.examples]
            data = {
                "examples": examples,
                "labelled": self.labelled,
                "predicted": self.predicted,
            }

            json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved PaiRagQcaDataset to {path}.")

    @classmethod
    def from_json(cls, path: str) -> "PaiRagQcaDataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)

        if len(data["examples"]) > 0:
            examples = [RagQcaSample.model_validate(el) for el in data["examples"]]
            labelled = data["labelled"]
            predicted = data["predicted"]

            return cls(examples=examples, labelled=labelled, predicted=predicted)
        else:
            return None
