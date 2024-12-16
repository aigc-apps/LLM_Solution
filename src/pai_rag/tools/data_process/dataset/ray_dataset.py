import os
import ray
import time
from pathlib import Path
from loguru import logger
from ray.data.datasource.filename_provider import _DefaultFilenameProvider
from pai_rag.tools.data_process.dataset.base_dataset import BaseDataset
from pai_rag.tools.data_process.utils.cuda_utils import get_num_gpus, calculate_np
import copy


class RayDataset(BaseDataset):
    def __init__(self, dataset_path: str = None, cfg=None) -> None:
        if os.path.isfile(dataset_path):
            self.data = ray.data.read_json(dataset_path)
        else:
            files = [
                str(file) for file in Path(dataset_path).rglob("*") if file.is_file()
            ]
            self.data = ray.data.read_json(files)
        self.num_proc = None
        if cfg:
            self.export_path = cfg.export_path

    def process(self, operators):
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]
        for op in operators:
            self._run_single_op(op)
            self.write_json(status=op._name)
        return self

    def _run_single_op(self, op):
        op_proc = calculate_np(
            op._name, op.mem_required, op.cpu_required, self.num_proc, op.use_cuda()
        )
        num_gpus = get_num_gpus(op.use_cuda(), op_proc)
        try:
            batch_size = getattr(op, "batch_size", 1)
            logger.info(
                f"Start processing with Op [{op._name}], batch_size: {batch_size}, op_proc: {op_proc}, num_gpus: {num_gpus}"
            )
            self.data = self.data.map_batches(
                op.process,
                batch_size=batch_size,
                num_cpus=op.cpu_required,
                num_gpus=num_gpus,
            )
        except:  # noqa: E722
            logger.error(f"An error occurred during Op [{op._name}].")
            import traceback

            traceback.print_exc()
            exit(1)

    def write_json(self, status):
        logger.info(f"Exporting {status} dataset to disk...")
        export_path = os.path.join(self.export_path, status)
        os.makedirs(export_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        export_data = copy.deepcopy(self.data)
        export_data.write_json(
            export_path,
            filename_provider=_DefaultFilenameProvider(
                dataset_uuid=timestamp, file_format="jsonl"
            ),
            force_ascii=False,
        )
        logger.info(f"Exported dataset to {export_path} completed.")
