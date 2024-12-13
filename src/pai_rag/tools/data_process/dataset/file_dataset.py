from pai_rag.tools.data_process.dataset.base_dataset import BaseDataset
from pai_rag.integrations.readers.pai.pai_data_reader import get_input_files
import ray
import json
from loguru import logger
import os
import time


class FileDataset(BaseDataset):
    def __init__(self, dataset_path: str = None, cfg=None) -> None:
        self.data = get_input_files(dataset_path)
        if cfg:
            self.export_path = cfg.export_path

    def process(self, operators):
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]
        for op in operators:
            self._run_single_op(op)
        return self

    def _run_single_op(self, op):
        try:
            run_tasks = [op.process.remote(input_file) for input_file in self.data]
            self.data = ray.get(run_tasks)
        except:  # noqa: E722
            logger.error(f"An error occurred during Op [{op._name}].")
            import traceback

            traceback.print_exc()
            exit(1)

    def write_json(self, export_path):
        if os.path.isdir(export_path):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            export_path = os.path.join(
                export_path, "pai_rag_parser", f"results_{timestamp}.jsonl"
            )
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
        with open(export_path, "a", encoding="utf-8") as f:
            for result in self.data:
                json_line = json.dumps(result, ensure_ascii=False)
                f.write(json_line + "\n")
