import os
import pytest
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.parent.parent


@pytest.mark.skipif(
    os.getenv("SKIP_GPU_TESTS", "false") == "true",
    reason="Need to execute in a CUDA environment.",
)
def test_markdown_parser():
    from pai_rag.core.rag_module import resolve
    from pai_rag.integrations.nodeparsers.pai.pai_markdown_parser import (
        MarkdownNodeParser,
    )
    from pai_rag.core.rag_config_manager import RagConfigManager
    from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
    from pai_rag.utils.download_models import ModelScopeDownloader

    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfigManager.from_file(config_file).get_value()
    directory_reader = resolve(
        cls=PaiDataReader,
        reader_config=config.data_reader,
    )
    input_dir = "tests/testdata/data/md_data"
    ModelScopeDownloader().load_rag_models()
    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    md_node_parser = MarkdownNodeParser(enable_multimodal=False)
    splitted_nodes = []
    for doc_node in documents:
        splitted_nodes.extend(md_node_parser.get_nodes_from_documents([doc_node]))

    text_list = [node.text for node in splitted_nodes]

    with open(
        "tests/testdata/data/json_data/pai_document.json", "r", encoding="utf-8"
    ) as file:
        chunk_text = json.load(file)

    assert text_list == chunk_text
    assert len(splitted_nodes) == 10
