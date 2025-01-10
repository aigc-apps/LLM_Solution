from typing import Optional
from loguru import logger

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import Settings
from llama_index.core.utilities.sql_wrapper import SQLDatabase

from pai_rag.integrations.data_analysis.data_analysis_config import (
    SqlAnalysisConfig,
)
from pai_rag.integrations.data_analysis.text2sql.db_info_collector import (
    SchemaCollector,
    HistoryCollector,
    ValueCollector,
)
from pai_rag.integrations.data_analysis.text2sql.db_info_node import (
    SchemaNode,
    HistoryNode,
    ValueNode,
)
from pai_rag.integrations.data_analysis.text2sql.db_info_retriever import (
    SchemaRetriever,
    HistoryRetriever,
    ValueRetriever,
)


class DBLoader:
    def __init__(
        self,
        db_config: SqlAnalysisConfig,
        sql_database: SQLDatabase,
        embed_model: BaseEmbedding,
        schema_retriever: Optional[SchemaRetriever] = None,
        history_retriever: Optional[HistoryRetriever] = None,
        value_retriever: Optional[ValueRetriever] = None,
        llm: Optional[LLM] = None,
    ):
        """Initialize offline process."""
        self._db_name = db_config.database
        self._embed_model = embed_model
        self._llm = llm or Settings.llm
        self._enable_db_history = db_config.enable_db_history
        self._enable_db_embedding = db_config.enable_db_embedding

        self._schema_retriever = schema_retriever
        self._history_retriever = history_retriever
        self._value_retriever = value_retriever

        self._schema_collector = SchemaCollector(
            db_name=self._db_name,
            sql_database=sql_database,
            context_query_kwargs=db_config.descriptions,
        )
        self._history_collector = HistoryCollector(db_name=self._db_name)
        self._value_collector = ValueCollector(
            db_name=self._db_name,
            sql_database=sql_database,
            max_col_num=db_config.max_col_num,
            max_val_num=db_config.max_val_num,
        )

        self._schema_node = SchemaNode(
            db_name=self._db_name, embed_model=self._embed_model
        )
        self._history_node = HistoryNode(
            db_name=self._db_name, embed_model=self._embed_model
        )
        self._value_node = ValueNode(
            db_name=self._db_name, embed_model=self._embed_model
        )

        logger.info("db_loader init successfully")

    def load_db_info(self):
        """
        处理数据库结构描述信息、历史查询和具体值信息, 存储json和索引
        """
        # get schema_description
        db_description_dict = self._schema_collector.collect()
        logger.info("db_description obtained.")

        # get db_history
        if self._enable_db_history:
            db_history_list = self._history_collector.collect()
            logger.info("db_history obtained.")
            history_nodes = self._history_node.create_nodes_with_embeddings(
                db_history_list
            )
            self._history_retriever.get_index(history_nodes)
            logger.info("db_history index stored.")

        # get db_embedding, including db_description, db_history, db_value
        if self._enable_db_embedding:
            description_nodes = self._schema_node.create_nodes_with_embeddings(
                db_description_dict
            )
            self._schema_retriever.get_index(description_nodes)
            logger.info("db_description index stored.")

            db_value_dict = self._value_collector.collect()
            value_nodes = self._value_node.create_nodes_with_embeddings(db_value_dict)
            self._value_retriever.get_index(value_nodes)
            logger.info("db_value index stored.")

    async def aload_db_info(self):
        """
        处理数据库结构描述信息、历史查询和具体值信息, 存储json和索引
        """
        # get schema_description
        db_description_dict = self._schema_collector.collect()
        logger.info("db_description obtained.")

        # get db_history
        if self._enable_db_history:
            db_history_list = self._history_collector.collect()
            logger.info("db_history obtained.")
            history_nodes = await self._history_node.acreate_nodes_with_embeddings(
                db_history_list
            )
            await self._history_retriever.aget_index(history_nodes)
            logger.info("db_history index stored.")

        # get db_embedding, including db_description, db_history, db_value
        if self._enable_db_embedding:
            description_nodes = await self._schema_node.acreate_nodes_with_embeddings(
                db_description_dict
            )
            await self._schema_retriever.aget_index(description_nodes)
            logger.info("db_description index stored.")

            db_value_dict = self._value_collector.collect()
            value_nodes = await self._value_node.acreate_nodes_with_embeddings(
                db_value_dict
            )
            await self._value_retriever.aget_index(value_nodes)
            logger.info("db_value index stored.")
