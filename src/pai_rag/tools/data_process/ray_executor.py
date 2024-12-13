import ray
import time
from loguru import logger
from pai_rag.tools.data_process.utils.op_utils import load_ops
from pai_rag.tools.data_process.dataset.ray_dataset import RayDataset
from pai_rag.tools.data_process.dataset.file_dataset import FileDataset


class RayExecutor:
    """
    Executor based on Ray.

    Run Data-Juicer data processing in a distributed cluster.

        1. Support Filter, Mapper and Exact Deduplicator operators for now.
        2. Only support loading `.json` files.
        3. Advanced functions such as checkpoint, tracer are not supported.

    """

    def __init__(self, cfg=None):
        """
        Initialization method.

        :param cfg: optional config dict.
        """
        self.cfg = cfg
        print("self.cfg", self.cfg)
        # init ray
        logger.info("Initing Ray ...")
        ray.init(runtime_env={"working_dir": self.cfg.working_dir})

    def run(self):
        """
        Running the dataset process pipeline.

        :param load_data_np: number of workers when loading the dataset.
        :return: processed dataset.
        """
        # 1. extract processes
        logger.info("Preparing process operators...")
        ops, op_names = load_ops(self.cfg.process)

        # 2. load data
        logger.info("Loading dataset with Ray...")
        if "pai_rag_parser" in op_names:
            idx = op_names.index("pai_rag_parser")
            op_names.pop(idx)
            parser_op = ops[idx]
            ops.pop(idx)
            dataset = FileDataset(self.cfg.dataset_path, self.cfg)
            # 3.1 data process - FileDataset
            logger.info("Processing file data...")
            tstart = time.time()
            dataset.process([parser_op])
            tend = time.time()
            logger.info(f"Op pai_rag_parser is done in {tend - tstart:.3f}s.")
            # 4.1 data export - FileDataset
            logger.info("Exporting dataset to disk...")
            dataset.write_json(self.cfg.export_path)
            self.cfg.dataset_path = self.cfg.export_path

        # convert all the path in dataset to absolute path
        dataset = RayDataset(self.cfg.dataset_path, self.cfg)
        # 3.2 data process - RayDataset
        logger.info("Processing ray data...")
        tstart = time.time()
        dataset.process(ops)
        tend = time.time()
        logger.info(f"All Ops are done in {tend - tstart:.3f}s.")

        # 4.2 data export - RayDataset
        logger.info("Exporting dataset to disk...")
        dataset.write_json(self.cfg.export_path, status="overall")
        return dataset
