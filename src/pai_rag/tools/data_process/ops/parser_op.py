import ray
from loguru import logger
from pai_rag.tools.data_process.ops.base_op import BaseOP, OPERATORS
from pai_rag.tools.data_process.utils.formatters import convert_document_to_dict
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.integrations.readers.pai.pai_data_reader import BaseDataReaderConfig
from pai_rag.core.rag_module import resolve
from pai_rag.utils.oss_client import OssClient
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
import threading

OP_NAME = "pai_rag_parser"


@OPERATORS.register_module(OP_NAME)
@ray.remote
class Parser(BaseOP):
    """Mapper to generate samples whose captions are generated based on
    another model and the figure."""

    _accelerator = "cpu"
    _batched_op = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.download_model_list = ["PDF-Extract-Kit"]
        self.load_models(self.download_model_list)
        logger.info("ParseActor init finished.")

    def load_models(self, model_list):
        download_models = ModelScopeDownloader(
            fetch_config=True,
            download_directory_path=self.model_dir,
        )
        for model_name in model_list:
            download_models.load_model(model=model_name)
        download_models.load_mineru_config(self.accelerator)

    def process(self, input_file):
        current_thread = threading.current_thread()
        import os

        logger.info(f"当前线程的 ID: {current_thread.ident} 进程ID: {os.getpid()}")

        self.data_reader_config = BaseDataReaderConfig()
        if self.kwargs.get("oss_store", None):
            self.oss_store = resolve(
                cls=OssClient,
                bucket_name=self.kwargs["oss_store"]["bucket"],
                endpoint=self.kwargs["oss_store"]["endpoint"],
            )
        else:
            self.oss_store = None

        data_reader = resolve(
            cls=PaiDataReader,
            reader_config=self.data_reader_config,
            oss_store=self.oss_store,
        )

        documents = data_reader.load_data(file_path_or_directory=input_file)
        if len(documents) == 0:
            logger.info(f"No data found in the input file: {input_file}")
            return None
        return convert_document_to_dict(documents[0])
