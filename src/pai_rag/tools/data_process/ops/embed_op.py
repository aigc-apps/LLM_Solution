import ray
import numpy as np
from pai_rag.tools.data_process.ops.base_op import BaseOP, OPERATORS
from pai_rag.integrations.index.pai.utils.sparse_embed_function import (
    BGEM3SparseEmbeddingFunction,
)
from pai_rag.utils.embed_utils import sync_download_url
from pai_rag.integrations.embeddings.pai.embedding_utils import create_embedding
from pai_rag.integrations.embeddings.pai.pai_embedding_config import parse_embed_config
from pai_rag.tools.data_process.utils.download_utils import download_models_via_lock


OP_NAME = "rag_embedder"


@OPERATORS.register_module(OP_NAME)
@ray.remote
class Embedder(BaseOP):
    """Mapper to generate samples whose captions are generated based on
    another model and the figure."""

    def __init__(
        self,
        source: str = None,
        model: str = None,
        enable_sparse: bool = False,
        enable_multimodal: bool = False,
        multimodal_source: str = None,
        connection_name: str = None,
        workspace_id: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedder_cfg = parse_embed_config(
            {
                "source": source,
                "model": model,
                "enable_sparse": enable_sparse,
                "connection_name": connection_name,
                "workspace_id": workspace_id,
            }
        )
        # Init model download list
        self.download_model_list = []
        if self.embedder_cfg.source.lower() == "huggingface":
            self.download_model_list.append(self.embedder_cfg.model)
        if self.embedder_cfg.enable_sparse:
            self.download_model_list.append("bge-m3")
        self.enable_multimodal = enable_multimodal
        if self.enable_multimodal:
            self.mm_embedder_cfg = parse_embed_config({"source": multimodal_source})
            self.download_model_list.append("chinese-clip-vit-large-patch14")
        for model_name in self.download_model_list:
            download_models_via_lock(self.model_dir, model_name)

        # Init embedding models
        self.embed_model = create_embedding(self.embedder_cfg)
        self.logger.info("Dense embedding model loaded.")
        if self.embedder_cfg.enable_sparse:
            self.sparse_embed_model = BGEM3SparseEmbeddingFunction(
                model_name_or_path=self.model_dir
            )
            self.logger.info("Sparse embedding model loaded.")
        if self.enable_multimodal:
            self.multimodal_embed_model = create_embedding(
                self.mm_embedder_cfg, pai_rag_model_dir=self.model_dir
            )
            self.logger.info("Multi-modal embedding model loaded.")
        self.logger.info(
            f"""EmbedderActor [PaiEmbedding] init finished with following parameters:
                        source: {source}
                        model: {model}
                        enable_sparse: {enable_sparse}
                        enable_multimodal: {enable_multimodal}
                        multimodal_source: {multimodal_source}
            """
        )

    def process_extra_metadata(self, nodes):
        excluded_embed_metadata_keys = nodes["excluded_embed_metadata_keys"]
        nodes["excluded_embed_metadata_keys"] = np.array(
            [list(a) for a in excluded_embed_metadata_keys]
        )
        excluded_llm_metadata_keys = nodes["excluded_llm_metadata_keys"]
        nodes["excluded_llm_metadata_keys"] = np.array(
            [list(a) for a in excluded_llm_metadata_keys]
        )
        nodes["start_char_idx"] = np.nan_to_num(nodes["start_char_idx"]).astype(int)
        nodes["end_char_idx"] = np.nan_to_num(nodes["start_char_idx"]).astype(int)
        return nodes

    def load_images_from_nodes(self, iamge_urls):
        results = [sync_download_url(url) for url in iamge_urls]
        return results

    def process(self, nodes):
        text_nodes = [node for node in nodes if node["type"] == "text"]
        image_nodes = [node for node in nodes if node["type"] == "image"]
        if len(text_nodes) > 0:
            text_contents = [node["text"] for node in text_nodes]
            embeddings = self.embed_model.get_text_embedding_batch(text_contents)
            if self.embedder_cfg.enable_sparse:
                sparse_embeddings = self.sparse_embed_model.encode_documents(
                    text_contents
                )
            else:
                sparse_embeddings = [None] * len(text_contents)
            # 回填embedding字段
            for node, embedding, sparse_embedding in zip(
                text_nodes, embeddings, sparse_embeddings
            ):
                node["embedding"] = embedding
                node["sparse_embedding"] = sparse_embedding
        else:
            self.logger.info("No text nodes to process.")

        if len(image_nodes) > 0:
            image_urls = [node["metadata"]["image_url"] for node in image_nodes]
            image_list = self.load_images_from_nodes(image_urls)
            image_embeddings = self.multimodal_embed_model.get_image_embedding_batch(
                image_list
            )
            for node, emb in zip(image_nodes, image_embeddings):
                node["embedding"] = emb
        else:
            self.logger.info("No image nodes to process.")

        return text_nodes + image_nodes
