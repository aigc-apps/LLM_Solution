import os
from typing import Any, Optional
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from pai_rag.utils.constants import DEFAULT_DASHSCOPE_EMBEDDING_MODEL
from pai_rag.integrations.embeddings.langstudio.langstudio_utils import (
    get_region_id,
    get_connection_info,
)
from loguru import logger


class LangStudioEmbedding:
    def __init__(
        self,
        region_id: Optional[str] = None,
        connection_name: Optional[str] = None,
        workspace_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embed_batch_size: Optional[int] = 8,
        **kwargs: Any,
    ):
        self.embedding_model = embedding_model
        self.embed_batch_size = embed_batch_size
        region_id = region_id or get_region_id()
        self.conn_info, self.configs, self.secrets = get_connection_info(
            region_id, connection_name, workspace_id
        )

    def get_embedding_model(self):
        if self.conn_info.custom_type == "OpenEmbeddingConnection":
            _model_name = self.embedding_model or "default"
            logger.info(
                f"Initializing LangStudioEmbedding with type: OpenEmbeddingConnection and model name: {_model_name}."
            )
            _api_key = self.secrets.get("api_key", None)
            _api_base = self.configs.get("base_url", None)
            return OpenAIEmbedding(
                model_name=_model_name,
                api_key=_api_key,
                api_base=_api_base,
                embed_batch_size=self.embed_batch_size,
            )
        elif self.conn_info.custom_type == "DashScopeConnection":
            _model_name = self.embedding_model or DEFAULT_DASHSCOPE_EMBEDDING_MODEL
            logger.info(
                f"Initializing LangStudioEmbedding with type: DashScopeConnection and model name: {_model_name}."
            )
            _api_key = self.secrets.get("api_key", None) or os.getenv(
                "DASHSCOPE_API_KEY"
            )
            return DashScopeEmbedding(
                api_key=_api_key,
                model_name=_model_name,
                embed_batch_size=self.embed_batch_size,
            )
