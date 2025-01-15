from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import json
import asyncio

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import QueryBundle

from pai_rag.integrations.data_analysis.data_analysis_config import SqliteAnalysisConfig
from pai_rag.integrations.data_analysis.text2sql.db_connector import SqliteConnector
from pai_rag.integrations.data_analysis.text2sql.db_info_retriever import (
    SchemaRetriever,
    ValueRetriever,
)
from pai_rag.integrations.data_analysis.text2sql.db_loader import DBLoader
from pai_rag.integrations.data_analysis.text2sql.db_query import DBQuery

from pai_rag.integrations.data_analysis.text2sql.evaluations.evaluation import (
    build_foreign_key_map_from_json,
    evaluate,
)
from pai_rag.integrations.data_analysis.data_analysis_config import SqlAnalysisConfig

cls_cache = {}


def resolve(cls: Any, cls_key: str, **kwargs):
    cls_key = kwargs.__repr__() + cls_key
    if cls_key not in cls_cache:
        cls_cache[cls_key] = cls(**kwargs)
        instance = cls(**kwargs)
        print(f"Created new instance with id: {id(instance)}")
    else:
        print(f"Returning cached instance with id: {id(cls_cache[cls_key])}")
    return cls_cache[cls_key]


def resolve_schema_retriever(
    analysis_config: SqlAnalysisConfig, embed_model: BaseEmbedding
):
    return resolve(
        cls=SchemaRetriever,
        cls_key="schema_retriever",
        db_name=analysis_config["db_name"],
        embed_model=embed_model,
        similarity_top_k=10,
    )


def resolve_value_retriever(
    analysis_config: SqlAnalysisConfig, embed_model: BaseEmbedding
):
    return resolve(
        cls=ValueRetriever,
        cls_key="value_retriever",
        db_name=analysis_config["db_name"],
        embed_model=embed_model,
        similarity_top_k=10,
    )


class SQLEvaluator(ABC):
    """生成SQL评估接口"""

    @abstractmethod
    async def abatch_loader(
        self,
    ):
        pass

    @abstractmethod
    async def abatch_query(self, nums: int):
        pass

    @abstractmethod
    def batch_evaluate(
        self, gold_file: str, predicted_file: str, evaluation_type: str, table_file: str
    ):
        pass


class SpiderEvaluator(SQLEvaluator):
    """SQLite数据库评估"""

    def __init__(
        self,
        llm: LLM,
        embed_model: BaseEmbedding,
        database_folder_path: str,
        eval_file_path: str,
        analysis_config: Dict,
    ):
        self._llm = llm
        self._embed_model = embed_model
        self._database_folder_path = database_folder_path
        self._sqlite_config_list: List[Dict[str, SqliteAnalysisConfig]] = []
        self._eval_file_path = eval_file_path

        database_folder = Path(database_folder_path)
        for db_file_path in tqdm(database_folder.rglob("*")):
            if db_file_path.is_file():
                if db_file_path.suffix.lower() == ".sqlite":
                    # prepare sqlite_config for each database file
                    db_name = db_file_path.stem
                    db_path = str(db_file_path.parent)
                    sqlite_config = SqliteAnalysisConfig(
                        db_path=db_path,
                        database=db_name,
                        enable_enhanced_description=analysis_config.get(
                            "enable_enhanced_description", False
                        ),
                        enable_db_history=analysis_config.get(
                            "enable_db_history", False
                        ),
                        enable_db_embedding=analysis_config.get(
                            "enable_db_embedding", False
                        ),
                        max_col_num=analysis_config.get("max_col_num", 100),
                        max_val_num=analysis_config.get("max_val_num", 10000),
                        enable_query_preprocessor=analysis_config.get(
                            "enable_query_preprocessor", False
                        ),
                        enable_db_preretriever=analysis_config.get(
                            "enable_db_preretriever", False
                        ),
                        enable_db_selector=analysis_config.get(
                            "enable_db_selector", False
                        ),
                    )
                    # connect each database and load its info
                    db_connector = SqliteConnector(sqlite_config)
                    sql_databse = db_connector.connect()
                    self._sqlite_config_list.append(
                        {
                            "db_name": db_name,
                            "sqlite_config": sqlite_config,
                            "sql_database": sql_databse,
                        }
                    )

    async def abatch_loader(
        self,
    ):
        tasks = []
        for config_item in self._sqlite_config_list:
            schema_retriever = resolve_schema_retriever(config_item, self._embed_model)
            value_retriever = resolve_value_retriever(config_item, self._embed_model)
            db_loader = DBLoader(
                db_config=config_item["sqlite_config"],
                sql_database=config_item["sql_database"],
                embed_model=self._embed_model,
                llm=self._llm,
                schema_retriever=schema_retriever,
                value_retriever=value_retriever,
            )
            # 创建任务列表
            tasks.append(db_loader.aload_db_info())

        # 并发运行所有任务
        await asyncio.gather(*tasks)

    async def abatch_query(self, nums: int):
        with open(self._eval_file_path, "r") as f:
            eval_list = json.load(f)
        tasks = []
        predicted_sql_list = []
        queried_result_list = []
        for eval_item in eval_list[0:nums]:
            # 创建任务列表
            tasks.append(
                self._aquery_eval_item(
                    eval_item, predicted_sql_list, queried_result_list
                )
            )

        # 并发运行所有任务
        await asyncio.gather(*tasks)

        return predicted_sql_list, queried_result_list

    async def _aquery_eval_item(
        self, eval_item, predicted_sql_list, queried_result_list
    ):
        for config_item in self._sqlite_config_list:
            if eval_item["db_id"] == config_item["db_name"]:
                schema_retriever = resolve_schema_retriever(
                    config_item, self._embed_model
                )
                value_retriever = resolve_value_retriever(
                    config_item, self._embed_model
                )
                db_query = DBQuery(
                    db_config=config_item["sqlite_config"],
                    sql_database=config_item["sql_database"],
                    embed_model=self._embed_model,
                    llm=self._llm,
                    schema_retriver=schema_retriever,
                    value_retriever=value_retriever,
                )
                query_bundle = QueryBundle(eval_item["question"])

                response_node, metadata = await db_query.aquery_pipeline(query_bundle)

                predicted_sql_list.append(
                    response_node[0].metadata["query_code_instruction"]
                )
                queried_result_list.append(response_node[0].metadata["query_output"])

    def parse_predicted_sql(self, predicted_sql_str: str):
        return predicted_sql_str.replace("\n", "")

    def batch_evaluate(
        self, gold_file: str, predicted_file: str, evaluation_type: str, table_file: str
    ):
        kmaps = build_foreign_key_map_from_json(table_file)

        evaluate(
            gold_file,
            predicted_file,
            self._database_folder_path,
            evaluation_type,
            kmaps,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
