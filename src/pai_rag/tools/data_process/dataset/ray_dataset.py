import os
import ray
import time
import json
from abc import ABC
from pathlib import Path
from loguru import logger
from pai_rag.tools.data_process.utils.formatters import NumpyEncoder


class RayDataset(ABC):
    def __init__(self, dataset_path: str = None, cfg=None) -> None:
        self.batch_size = 10
        logger.info(f"Loading json dataset from {dataset_path}.")
        if os.path.isfile(dataset_path):
            self.data = self.read_jsonl_in_batches([dataset_path])
        else:
            files = [
                str(file) for file in Path(dataset_path).rglob("*") if file.is_file()
            ]
            self.data = self.read_jsonl_in_batches(files)
        self.num_proc = None
        if cfg:
            self.export_path = cfg.export_path

    def read_jsonl_in_batches(self, files):
        for file_path in files:
            logger.info(f"Start loading data from {file_path}")
            first_line = open(file_path, "r").readline()
            logger.info(f"First line: {first_line}")
            batch_count, line_count = 0, 0
            with open(file_path, "r") as file:
                batch = []
                for line in file:
                    line_count += 1
                    batch.append(json.loads(line))
                    if len(batch) >= self.batch_size:
                        batch_count += 1
                        yield batch
                        batch = []
                if batch:
                    batch_count += 1
                    yield batch
            logger.info(
                f"Finish loading data from {file_path}: batch count: {batch_count} line count: {line_count}"
            )

    def process(self, operators, op_name):
        self._run_single_op(operators, op_name)
        self.write_json(status=op_name)

    def _run_single_op(self, ops, op_name):
        try:
            logger.info(f"Running Op [{op_name}] with {len(ops)} actors.")
            num_actors = len(ops)
            run_tasks = []
            for i, batch_data in enumerate(self.data):
                run_tasks.append(ops[i % num_actors].process.remote(batch_data))
            self.data = ray.get(run_tasks)
        except:  # noqa: E722
            import traceback

            logger.error(
                f"An error occurred during Op [{op_name} {traceback.print_exc()}]."
            )
            exit(1)

    def write_json(self, status):
        logger.info(f"Exporting {status} dataset to disk...")
        export_path = os.path.join(self.export_path, status)
        os.makedirs(export_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        export_file_path = os.path.join(export_path, f"results_{timestamp}.jsonl")
        with open(export_file_path, "w") as f:
            for result in self.data:
                for line in result:
                    json_line = json.dumps(line, ensure_ascii=False, cls=NumpyEncoder)
                    f.write(json_line + "\n")
        logger.info(f"Exported dataset to {export_file_path} completed.")
