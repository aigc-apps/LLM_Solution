import os
from dotenv import load_dotenv

from llama_index.core.schema import QueryBundle
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.core import Settings
from pai_rag.integrations.data_analysis.text2sql.query_processor import KeywordExtractor

# 加载 .env 文件中的环境变量
load_dotenv()

dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_TURBO,
    api_key=dashscope_api_key,
    temperature=0.1,
    max_tokens=2048,
)
Settings.llm = llm
# query = "有猫的学生有多少？"
# qp = KeywordExtraction()
# result = qp.process(QueryBundle(query))
# print(result)


def test_query_processor():
    query = "有猫的学生有多少？"
    qp = KeywordExtractor()
    keywords = qp.process(QueryBundle(query))
    assert isinstance(keywords, list)
    assert len(keywords) > 0
    assert "猫" in keywords
    assert "学生" in keywords
