from abc import ABC, abstractmethod
from typing import List

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.schema import QueryBundle

from pai_rag.integrations.data_analysis.text2sql.db_info_index import (
    HistoryIndex,
    ValueIndex,
    SchemaIndex,
)


# 数据库信息（向量）检索接口
class DBInfoRetriever(ABC):
    """
    Abstract base class for database information vector-based retrieval.
    """

    def __init__(self, db_name: str, embed_model: BaseEmbedding, similarity_top_k: int):
        self._db_name = db_name
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k

    @abstractmethod
    def get_index(self, nodes: List[TextNode]):
        pass

    @abstractmethod
    async def aget_index(self, nodes: List[TextNode]):
        pass

    @abstractmethod
    def retrieve_nodes(self, query: QueryBundle | str | List):
        pass

    @abstractmethod
    async def aretrieve_nodes(self, query: QueryBundle | str | List):
        pass


class SchemaRetriever(DBInfoRetriever):
    def __init__(self, db_name: str, embed_model: BaseEmbedding, similarity_top_k: int):
        super().__init__(db_name, embed_model, similarity_top_k)
        # self._schema_nodes = SchemaNode(embed_model=embed_model)
        self._schema_index = SchemaIndex(
            db_name=db_name, embed_model=embed_model, similarity_top_k=similarity_top_k
        )
        self._schema_retriever = self._schema_index.as_retriever()

    def get_index(self, nodes: List[TextNode]):
        # nodes = self._schema_nodes.create_nodes_with_embeddings()
        self._schema_index.insert_nodes(nodes)

    async def aget_index(self, nodes: List[TextNode]):
        # nodes = self._schema_nodes.acreate_nodes_with_embeddings()
        self._schema_index.insert_nodes(nodes)

    def retrieve_nodes(self, query):
        retrieved_nodes = self._schema_retriever.retrieve(query)

        return retrieved_nodes

    async def aretrieve_nodes(self, query):
        retrieved_nodes = await self._schema_retriever.aretrieve(query)

        return retrieved_nodes


class HistoryRetriever(DBInfoRetriever):
    def __init__(self, db_name: str, embed_model: BaseEmbedding, similarity_top_k: int):
        super().__init__(db_name, embed_model, similarity_top_k)
        # self._history_nodes = HistoryNode(embed_model=embed_model)
        self._history_index = HistoryIndex(
            db_name=db_name, embed_model=embed_model, similarity_top_k=similarity_top_k
        )
        self._history_retriever = self._history_index.as_retriever()

    def get_index(self, nodes: List[TextNode]):
        # nodes = self._history_nodes.create_nodes_with_embeddings()
        self._history_index.insert_nodes(nodes)

    async def aget_index(self, nodes: List[TextNode]):
        # nodes = self._history_nodes.acreate_nodes_with_embeddings()
        self._history_index.insert_nodes(nodes)

    def retrieve_nodes(self, query):
        retrieved_nodes = self._history_retriever.retrieve(query)

        return retrieved_nodes

    async def aretrieve_nodes(self, query):
        retrieved_nodes = await self._history_retriever.aretrieve(query)

        return retrieved_nodes


class ValueRetriever(DBInfoRetriever):
    def __init__(self, db_name: str, embed_model: BaseEmbedding, similarity_top_k: int):
        super().__init__(db_name, embed_model, similarity_top_k)
        # self._value_nodes = ValueNode(embed_model=embed_model)
        self._value_index = ValueIndex(
            db_name=db_name, embed_model=embed_model, similarity_top_k=similarity_top_k
        )
        self._value_retriever = self._value_index.as_retriever()

    def get_index(self, nodes: List[TextNode]):
        # nodes = self._value_nodes.create_nodes_with_embeddings()
        self._value_index.insert_nodes(nodes)

    async def aget_index(self, nodes: List[TextNode]):
        # nodes = self._value_nodes.acreate_nodes_with_embeddings()
        self._value_index.insert_nodes(nodes)

    def retrieve_nodes(self, query):
        retrieved_nodes_list = []
        if len(query) != 0:
            for keyword in query:
                retrieved_nodes = self._value_retriever.retrieve(keyword)
                retrieved_nodes_list.extend(retrieved_nodes)

        return retrieved_nodes_list

    async def aretrieve_nodes(self, query):
        retrieved_nodes_list = []
        if len(query) != 0:
            for keyword in query:
                retrieved_nodes = await self._value_retriever.aretrieve(keyword)
                retrieved_nodes_list.extend(retrieved_nodes)

        return retrieved_nodes_list
