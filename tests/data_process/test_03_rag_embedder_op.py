import os
from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).parent.parent.parent


def test_rag_embedder_op():
    # 配置测试用的参数
    operator = "rag_embedder"
    working_dir = BASE_DIR
    dataset_path = os.path.join(BASE_DIR, "tests/data_process/output")
    export_path = os.path.join(BASE_DIR, "tests/data_process/output")

    # 执行命令行程序
    command = [
        sys.executable,
        "src/pai_rag/tools/data_process/run.py",
        "--operator",
        operator,
        "--working_dir",
        working_dir,
        "--dataset_path",
        dataset_path,
        "--export_path",
        export_path,
        "--source",
        "dashscope",
        "--enable_sparse",
        "True",
    ]
    # add the correct value for the PYTHONPATH when invoking the subprocess call when using the subprocess module in the context of GitHub's CI
    current_env = os.environ.copy()
    subprocess.run(command, capture_output=True, text=True, env=current_env)

    embedder_export_path = Path(os.path.join(export_path, "rag_embedder"))

    # 遍历导出目录下的所有文件
    for file_path in embedder_export_path.iterdir():
        if file_path.is_file():
            with open(file_path, "r") as file:
                assert len(file.readlines()) == 9
