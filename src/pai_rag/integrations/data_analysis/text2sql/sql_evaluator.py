from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import json
import asyncio
from loguru import logger

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
from pai_rag.integrations.data_analysis.text2sql.evaluations.evaluation_bird import (
    package_sqls,
    print_data,
    # result_callback,
    run_sqls_parallel,
    sort_results,
    compute_acc_by_diff,
)
from pai_rag.integrations.data_analysis.data_analysis_config import SqlAnalysisConfig

cls_cache = {}


def resolve(cls: Any, cls_key: str, **kwargs):
    cls_key = kwargs.__repr__() + cls_key
    if cls_key not in cls_cache:
        cls_cache[cls_key] = cls(**kwargs)
        instance = cls(**kwargs)
        logger.debug(f"Created new instance with id: {id(instance)}")
    else:
        logger.debug(f"Returning cached instance with id: {id(cls_cache[cls_key])}")
    return cls_cache[cls_key]


def resolve_schema_retriever(
    analysis_config: SqlAnalysisConfig, embed_model: BaseEmbedding
):
    return resolve(
        cls=SchemaRetriever,
        cls_key="schema_retriever",
        db_name=analysis_config["db_name"],
        embed_model=embed_model,
        similarity_top_k=5,
    )


def resolve_value_retriever(
    analysis_config: SqlAnalysisConfig, embed_model: BaseEmbedding
):
    return resolve(
        cls=ValueRetriever,
        cls_key="value_retriever",
        db_name=analysis_config["db_name"],
        embed_model=embed_model,
        similarity_top_k=5,
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
    """公开数据集SPIDER SQLite数据库评估"""

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

        # # 控制并发任务数量
        # semaphore = asyncio.Semaphore(20)

        # async def limited_load(config_item):
        #     async with semaphore:
        #         try:
        #             await self._aload_config_item(config_item)
        #             logger.info(
        #                 f"""Successfully loaded database {config_item["sql_database"]}"""
        #             )
        #         except Exception as e:
        #             logger.error(
        #                 f"""Error loaded database {config_item["sql_database"]}: {e}"""
        #             )

        # tasks = [limited_load(config_item) for config_item in self._sqlite_config_list]

        # await asyncio.gather(*tasks)

    # async def _aload_config_item(self, config_item):
    #     for config_item in self._sqlite_config_list:
    #         schema_retriever = resolve_schema_retriever(config_item, self._embed_model)
    #         value_retriever = resolve_value_retriever(config_item, self._embed_model)
    #         db_loader = DBLoader(
    #             db_config=config_item["sqlite_config"],
    #             sql_database=config_item["sql_database"],
    #             embed_model=self._embed_model,
    #             llm=self._llm,
    #             schema_retriever=schema_retriever,
    #             value_retriever=value_retriever,
    #         )
    #         await db_loader.aload_db_info()

    async def abatch_query(self, nums: int):
        with open(self._eval_file_path, "r") as f:
            eval_list = json.load(f)

        # 控制并发任务数量
        semaphore = asyncio.Semaphore(20)

        async def limited_query(index, eval_item):
            async with semaphore:
                try:
                    result = await self._aquery_eval_item(index, eval_item)
                    logger.info(f"Successfully processed query {index}")
                    return result
                except Exception as e:
                    logger.error(f"Error processing query {index}: {e}")
                    return (index, None, None)  # 返回默认值或特殊标记

        tasks = [
            limited_query(i, eval_item) for i, eval_item in enumerate(eval_list[0:nums])
        ]

        # # 创建带有索引的任务列表
        # tasks = [
        #     self._aquery_eval_item(i, eval_item)
        #     for i, eval_item in enumerate(eval_list[0:nums])
        # ]

        # 并发运行所有任务并收集带索引的结果
        results = await asyncio.gather(*tasks)

        # 按索引排序结果
        sorted_results = sorted(results, key=lambda x: x[0])

        # 分离预测的 SQL 列表和查询结果列表
        predicted_sql_list = [item[1] for item in sorted_results]
        queried_result_list = [item[2] for item in sorted_results]

        return predicted_sql_list, queried_result_list

    async def _aquery_eval_item(self, idx, eval_item):
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
                    schema_retriever=schema_retriever,
                    value_retriever=value_retriever,
                )
                query_bundle = QueryBundle(eval_item["question"])

                response_node, metadata = await db_query.aquery_pipeline(query_bundle)
                # predicted_sql_list.append(
                #     response_node[0].metadata["query_code_instruction"]
                # )
                # queried_result_list.append(response_node[0].metadata["query_output"])
                return (
                    idx,
                    response_node[0].metadata["query_code_instruction"],
                    response_node[0].metadata["query_output"],
                )

    def batch_query(self, nums: int):
        with open(self._eval_file_path, "r") as f:
            eval_list = json.load(f)

        predicted_sql_list = []
        queried_result_list = []
        for eval_item in eval_list[0:nums]:
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
                        schema_retriever=schema_retriever,
                        value_retriever=value_retriever,
                    )
                    query_bundle = QueryBundle(eval_item["question"])

                    response_node, metadata = db_query.query_pipeline(query_bundle)
                    predicted_sql_list.append(
                        response_node[0].metadata["query_code_instruction"]
                    )
                    queried_result_list.append(
                        response_node[0].metadata["query_output"]
                    )

        return predicted_sql_list, queried_result_list

    def parse_predicted_sql(self, predicted_sql_str: str):
        if "\n" in predicted_sql_str:
            return predicted_sql_str.replace("\n", "")
        else:
            return predicted_sql_str

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


class BirdEvaluator(SQLEvaluator):
    """公开数据集BIRD SQLite数据库评估"""

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
        # # 控制并发任务数量
        # semaphore = asyncio.Semaphore(10)

        # async def limited_load(config_item):
        #     async with semaphore:
        #         try:
        #             await self._aload_config_item(config_item)
        #             logger.info(
        #                 f"""Successfully loaded database {config_item["db_name"]}"""
        #             )
        #         except Exception as e:
        #             logger.error(
        #                 f"""Error loaded database {config_item["db_name"]}: {e}"""
        #             )

        # tasks = [limited_load(config_item) for config_item in self._sqlite_config_list]

        # await asyncio.gather(*tasks)

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
                database_file_path=self._database_folder_path,
            )
            # 创建任务列表
            tasks.append(db_loader.aload_db_info())

        # 并发运行所有任务
        await asyncio.gather(*tasks)

    async def _aload_config_item(self, config_item):
        for config_item in self._sqlite_config_list:
            # print("aload_config_item:", config_item["db_name"])
            schema_retriever = resolve_schema_retriever(config_item, self._embed_model)
            value_retriever = resolve_value_retriever(config_item, self._embed_model)
            db_loader = DBLoader(
                db_config=config_item["sqlite_config"],
                sql_database=config_item["sql_database"],
                embed_model=self._embed_model,
                llm=self._llm,
                schema_retriever=schema_retriever,
                value_retriever=value_retriever,
                database_file_path=self._database_folder_path,
            )
            await db_loader.aload_db_info()

    async def abatch_query(self, nums: int):
        with open(self._eval_file_path, "r") as f:
            eval_list = json.load(f)

        # 控制并发任务数量
        semaphore = asyncio.Semaphore(20)

        async def limited_query(index, eval_item):
            async with semaphore:
                try:
                    result = await self._aquery_eval_item(index, eval_item)
                    logger.info(f"Successfully processed query {index}")
                    return result
                except Exception as e:
                    logger.error(f"Error processing query {index}: {e}")
                    return (index, None, None, None)  # 返回默认值或特殊标记

        tasks = [
            limited_query(i, eval_item) for i, eval_item in enumerate(eval_list[0:nums])
        ]
        # 并发运行所有任务并收集带索引的结果
        results = await asyncio.gather(*tasks)
        # 按索引排序结果
        sorted_results = sorted(results, key=lambda x: x[0])

        # 分离预测的 SQL 列表和查询结果列表
        db_id_list = [item[1] for item in sorted_results]
        predicted_sql_list = [item[2] for item in sorted_results]
        queried_result_list = [item[3] for item in sorted_results]

        return predicted_sql_list, queried_result_list, db_id_list

    async def _aquery_eval_item(self, idx, eval_item):
        for config_item in self._sqlite_config_list:
            if eval_item["db_id"] == config_item["db_name"]:
                # print("aquery_eval_item:", eval_item["db_id"])
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
                    schema_retriever=schema_retriever,
                    value_retriever=value_retriever,
                )
                query_bundle = QueryBundle(eval_item["question"])
                query_hint = eval_item["evidence"]
                logger.info(f"nl_query: {query_bundle}, query_hint: {query_hint}")

                response_node, metadata = await db_query.aquery_pipeline(
                    query_bundle, query_hint
                )

                return (
                    idx,
                    config_item["db_name"],
                    response_node[0].metadata["query_code_instruction"],
                    response_node[0].metadata["query_output"],
                )

    def parse_predicted_sql(self, predicted_sql_str: str):
        if "\n" in predicted_sql_str:
            return predicted_sql_str.replace("\n", " ")
        else:
            return predicted_sql_str

    def batch_evaluate(
        self,
        predicted_sql_path: str,
        ground_truth_path: str,
        db_root_path: str,
        data_mode: str = "dev",
        num_cpus: int = 1,
        meta_time_out: float = 30.0,
        mode_gt: str = "gt",
        mode_predict: str = "gpt",
        difficulty: str = "simple",
        diff_json_path: str = "",
        exec_result: list = [],
    ):
        pred_queries, db_paths = package_sqls(
            predicted_sql_path, db_root_path, mode=mode_predict, data_mode=data_mode
        )
        # generate gt sqls:
        gt_queries, db_paths_gt = package_sqls(
            ground_truth_path, db_root_path, mode="gt", data_mode=data_mode
        )

        query_pairs = list(zip(pred_queries, gt_queries))
        run_sqls_parallel(
            query_pairs,
            db_places=db_paths,
            exec_result=exec_result,
            num_cpus=num_cpus,
            meta_time_out=meta_time_out,
        )
        exec_result = sort_results(exec_result)

        print("start calculate")
        (
            simple_acc,
            moderate_acc,
            challenging_acc,
            acc,
            count_lists,
        ) = compute_acc_by_diff(exec_result, diff_json_path)
        score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
        print_data(score_lists, count_lists)
        print(
            "==========================================================================================="
        )
        print("Finished evaluation")
