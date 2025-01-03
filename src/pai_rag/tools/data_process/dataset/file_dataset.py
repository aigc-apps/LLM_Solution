import ray
import json
import os
import time
from abc import ABC
from loguru import logger
from pai_rag.integrations.readers.pai.pai_data_reader import get_input_files


class FileDataset(ABC):
    def __init__(self, dataset_path: str = None, cfg=None) -> None:
        logger.info(f"Loading file dataset from {dataset_path}.")
        self.data, _ = get_input_files(dataset_path)
        if cfg:
            self.export_path = cfg.export_path

    def process(self, operators, op_name):
        self._run_single_op(operators, op_name)
        self.write_json(status=op_name)

    def _run_single_op(self, ops, op_name):
        try:
            logger.info(f"Running Op [{op_name}]")
            num_actors = len(ops)
            run_tasks = []
            for i, batch_data in enumerate(self.data):
                run_tasks.append(ops[i % num_actors].process.remote(batch_data))
            # run_tasks = [ops.process.remote(batch_data) for batch_data in self.data]
            self.data = ray.get(run_tasks)
        except:  # noqa: E722
            logger.error(f"An error occurred during Op [{op_name}].")
            import traceback

            traceback.print_exc()
            exit(1)

    def write_json(self, status):
        logger.info(f"Exporting {status} dataset to disk...")
        export_path = os.path.join(self.export_path, status)
        os.makedirs(export_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        export_file_path = os.path.join(export_path, f"results_{timestamp}.jsonl")
        with open(export_file_path, "w") as f:
            for result in self.data:
                json_line = json.dumps(result, ensure_ascii=False)
                f.write(json_line + "\n")
        logger.info(f"Exported dataset to {export_file_path} completed.")
