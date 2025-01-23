import os
import asyncio
from dotenv import load_dotenv
from loguru import logger
import sys
import pickle

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding

from pai_rag.integrations.data_analysis.text2sql.sql_evaluator import (
    SpiderEvaluator,
)

# 移除默认的日志处理器
logger.remove()
# 添加一个新的日志处理器，指定最低日志级别为 INFO，并输出到指定文件
log_file_path = "/Users/chuyu/Documents/rag_doc/text2sql_evaluation/spider/spider_eval.log"  # 指定日志文件路径
logger.add(
    log_file_path, level="DEBUG", rotation="50 MB", retention="10 days", enqueue=True
)
# 添加一个可选的日志处理器，输出到标准错误（如果你仍然希望在控制台看到日志）
logger.add(sys.stderr, level="INFO")


# 加载 .env 文件中的环境变量
load_dotenv()

# 获取环境变量中的 API 密钥
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

if os.path.exists("./model_repository/bge-m3"):
    embed_model_bge = HuggingFaceEmbedding(model_name="./model_repository/bge-m3")
else:
    embed_model_bge = None

llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,
    api_key=dashscope_api_key,
    temperature=0.1,
    max_tokens=4096,
)
embed_model_dashscope = DashScopeEmbedding(
    api_key=dashscope_api_key,
    embed_batch_size=10,
)


if __name__ == "__main__":
    # database_folder_path = "/Users/chuyu/Documents/datasets/spider_data/test_database"
    database_folder_path = "/Users/chuyu/Documents/datasets/temp_test"
    eval_file_path = "/Users/chuyu/Documents/datasets/spider_data/test.json"
    history_file_path = "/Users/chuyu/Documents/datasets/spider_data/train_spider.json"
    analysis_config = {
        "enable_enhanced_description": False,
        "enable_db_history": True,
        "enable_db_embedding": True,
        "max_col_num": 100,
        "max_val_num": 10000,
        "enable_query_preprocessor": True,
        "enable_db_preretriever": True,
        "enable_db_selector": False,
    }

    spider_eval = SpiderEvaluator(
        llm=llm,
        embed_model=embed_model_bge,
        database_folder_path=database_folder_path,
        eval_file_path=eval_file_path,
        analysis_config=analysis_config,
        history_file_path=history_file_path,
    )

    # batch_load
    asyncio.run(spider_eval.abatch_loader())

    # batch_predict
    predicted_sql_list, queried_result_list = asyncio.run(
        spider_eval.abatch_query(nums=100)
    )

    # 写入二进制文件
    with open(
        "/Users/chuyu/Documents/rag_doc/text2sql_evaluation/spider/predicted_sql_list.pkl",
        "wb",
    ) as file:
        pickle.dump(predicted_sql_list, file)

    # save result
    predicted_file = (
        "/Users/chuyu/Documents/rag_doc/text2sql_evaluation/spider/predict_qwenmax.txt"
    )
    with open(predicted_file, "w", encoding="utf-8") as file:
        for item in predicted_sql_list:
            if item:
                parsed_item = spider_eval.parse_predicted_sql(item)
            file.write(f"{parsed_item}\n")
    print(f"predicted_sql_list have been written to {predicted_file}")

    # batch_evaluate
    gold_file = "/Users/chuyu/Documents/datasets/spider_data/test_gold.sql"
    table_file = "/Users/chuyu/Documents/datasets/spider_data/test_tables.json"

    spider_eval.batch_evaluate(
        gold_file=gold_file,
        predicted_file=predicted_file,
        evaluation_type="exec",
        table_file=table_file,
    )
