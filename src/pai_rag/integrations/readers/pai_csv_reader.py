"""Tabular parser-CSV parser.

Contains parsers for tabular data files.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem

import pandas as pd
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PaiCSVReader(BaseReader):
    """CSV parser.

    Args:
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.
        csv_config （dict）: Options for the reader.Set to empty dict by default,
          this means reader will try to figure
            out the separators, table head, etc. on its own.

    """

    def __init__(
        self, *args: Any, concat_rows: bool = True, csv_config: dict = {}, **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._csv_config = csv_config

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse csv file.

        Returns:
            Union[str, List[str]]: a string or a List of strings.

        """
        try:
            import csv
        except ImportError:
            raise ImportError("csv module is required to read CSV files.")
        text_list = []
        headers = []
        data_lines = []
        data_line_start_index = 1
        if (
            "header" in self._csv_config
            and self._csv_config["header"] is not None
            and isinstance(self._csv_config["header"], list)
        ):
            data_line_start_index = max(self._csv_config["header"]) + 1
        elif (
            "header" in self._csv_config
            and self._csv_config["header"] is not None
            and isinstance(self._csv_config["header"], int)
        ):
            data_line_start_index = self._csv_config["header"] + 1
            self._csv_config["header"] = [self._csv_config["header"]]

        with open(file) as fp:
            has_header = csv.Sniffer().has_header(fp.read(2048))
            fp.seek(0)

            if "header" not in self._csv_config and has_header:
                self._csv_config["header"] = [0]
            elif "header" not in self._csv_config and not has_header:
                self._csv_config["header"] = None

            csv_reader = csv.reader(fp)

            if self._csv_config["header"] is None:
                for row in csv_reader:
                    text_list.append(", ".join(row))
            else:
                for i, row in enumerate(csv_reader):
                    if i in self._csv_config["header"]:
                        headers.append(row)
                    elif i >= data_line_start_index:
                        data_lines.append(row)
                headers = [tuple(group) for group in zip(*headers)]
                for line in data_lines:
                    if len(line) == len(headers):
                        data_entry = str(dict(zip(headers, line)))
                        text_list.append(data_entry)

        metadata = {"filename": file.name, "extension": file.suffix}
        if extra_info:
            metadata = {**metadata, **extra_info}

        if self._concat_rows:
            return [Document(text="\n".join(text_list), metadata=metadata)]
        else:
            return [Document(text=text, metadata=metadata) for text in text_list]


class PaiPandasCSVReader(BaseReader):
    r"""Pandas-based CSV parser.

    Parses CSVs using the separator detection from Pandas `read_csv`function.
    If special parameters are required, use the `pandas_config` dict.

    Args:
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.

        row_joiner (str): Separator to use for joining each row.
            Only used when `concat_rows=True`.
            Set to "\n" by default.

        pandas_config (dict): Options for the `pandas.read_csv` function call.
            Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
            for more information.
            Set to empty dict by default, this means pandas will try to figure
            out the separators, table head, etc. on its own.

    """

    def __init__(
        self,
        *args: Any,
        concat_rows: bool = True,
        row_joiner: str = "\n",
        pandas_config: dict = {},
        **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse csv file."""
        if fs:
            with fs.open(file) as f:
                df = pd.read_csv(f, **self._pandas_config)
        else:
            df = pd.read_csv(file, **self._pandas_config)

        text_list = df.apply(
            lambda row: str(dict(zip(df.columns, row.astype(str)))), axis=1
        ).tolist()

        if self._concat_rows:
            return [
                Document(
                    text=(self._row_joiner).join(text_list), metadata=extra_info or {}
                )
            ]
        else:
            return [
                Document(text=text, metadata=extra_info or {}) for text in text_list
            ]
