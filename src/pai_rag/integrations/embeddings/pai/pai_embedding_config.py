from typing import Literal
from pydantic import BaseModel, Field
from enum import Enum
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
import os

DEFAULT_HF_EMBED_MODEL = "bge-m3"


class SupportedEmbedType(str, Enum):
    dashscope = "dashscope"
    openai = "openai"
    huggingface = "huggingface"
    cnclip = "cnclip"  # Chinese CLIP


class PaiBaseEmbeddingConfig(BaseModel):
    source: SupportedEmbedType
    model: str
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    enable_sparse: bool = False

    class Config:
        frozen = True

    @classmethod
    def get_subclasses(cls):
        return tuple(cls.__subclasses__())

    @classmethod
    def get_type(cls):
        return cls.model_fields["source"].default


class DashScopeEmbeddingConfig(PaiBaseEmbeddingConfig):
    source: Literal[SupportedEmbedType.dashscope] = SupportedEmbedType.dashscope
    model: str | None = None  # use default
    api_key: str | None = Field(default=os.getenv("DASHSCOPE_API_KEY"))  # use default


class OpenAIEmbeddingConfig(PaiBaseEmbeddingConfig):
    source: Literal[SupportedEmbedType.openai] = SupportedEmbedType.openai
    model: str | None = None  # use default
    api_key: str | None = None  # use default


class HuggingFaceEmbeddingConfig(PaiBaseEmbeddingConfig):
    source: Literal[SupportedEmbedType.huggingface] = SupportedEmbedType.huggingface
    model: str | None = DEFAULT_HF_EMBED_MODEL


class CnClipEmbeddingConfig(PaiBaseEmbeddingConfig):
    source: Literal[SupportedEmbedType.cnclip] = SupportedEmbedType.cnclip
    model: str | None = "ViT-L-14"


SupporttedEmbeddingClsMap = {
    cls.get_type(): cls for cls in PaiBaseEmbeddingConfig.get_subclasses()
}


def parse_embed_config(config_data):
    if "source" not in config_data:
        raise ValueError("Embedding config must contain 'source' field")

    embedding_cls = SupporttedEmbeddingClsMap.get(config_data["source"].lower())
    if embedding_cls is None:
        raise ValueError(f"Unsupported embedding source: {config_data['source']}")

    return embedding_cls(**config_data)


if __name__ == "__main__":
    embedding_config_data = {"source": "Openai", "model": "gpt-1", "api_key": None}

    print(parse_embed_config(embedding_config_data))
