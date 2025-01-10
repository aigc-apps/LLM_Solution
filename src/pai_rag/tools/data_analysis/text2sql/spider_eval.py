import os
import asyncio
from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

from pai_rag.integrations.data_analysis.text2sql.sql_evaluator import (
    SpiderEvaluator,
)

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
    max_tokens=2048,
)


if __name__ == "__main__":
    database_folder_path = "/Users/chuyu/Documents/datasets/spider_data/test_database"
    eval_file_path = "/Users/chuyu/Documents/datasets/spider_data/test.json"

    spider_eval = SpiderEvaluator(
        llm=llm,
        embed_model=embed_model_bge,
        database_folder_path=database_folder_path,
        eval_file_path=eval_file_path,
    )

    asyncio.run(spider_eval.abatch_loader())

    predicted_sql_list, queried_result_list = asyncio.run(
        spider_eval.abatch_query(nums=100)
    )

    # 文件路径
    file_path = "/Users/chuyu/Documents/predict_qwenmax_100.txt"

    # 写入文件，每行一个元素并使用对象的 __str__ 方法
    with open(file_path, "w", encoding="utf-8") as file:
        for item in predicted_sql_list:
            file.write(f"{item}\n")

    print(f"predicted_sql_list have been written to {file_path}")

    gold_file = "/Users/chuyu/Documents/gold_100.sql"
    predicted_file = "/Users/chuyu/Documents/predict_qwenmax_100.txt"
    table_file = "/Users/chuyu/Documents/datasets/spider_data/test_tables.json"

    spider_eval.batch_evaluate(
        gold_file=gold_file,
        predicted_file=predicted_file,
        evaluation_type="all",
        table_file=table_file,
    )
