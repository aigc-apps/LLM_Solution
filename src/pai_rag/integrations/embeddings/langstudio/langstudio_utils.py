import os
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_credentials.models import Config as CredentialConfig
from alibabacloud_pailangstudio20240710.client import Client as LangStudioClient
from alibabacloud_pailangstudio20240710.models import (
    GetConnectionRequest,
    ListConnectionsRequest,
)
from alibabacloud_tea_openapi import models as open_api_models
from loguru import logger


def get_region_id():
    return next(
        (
            os.environ[key]
            for key in ["REGION", "REGION_ID", "ALIBABA_CLOUD_REGION_ID"]
            if key in os.environ and os.environ[key]
        ),
        "cn-hangzhou",
    )


def get_connection_info(region_id: str, connection_name: str, workspace_id: str):
    """
    Get Connection information from LangStudio API.
    """
    config1 = CredentialConfig(
        type="access_key",
        access_key_id=os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID"),
        access_key_secret=os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
    )
    public_endpoint = f"pailangstudio.{region_id}.aliyuncs.com"
    client = LangStudioClient(
        config=open_api_models.Config(
            # Use default credential chain, see:
            # https://help.aliyun.com/zh/sdk/developer-reference/v2-manage-python-access-credentials#3ca299f04bw3c
            credential=CredentialClient(config=config1),
            endpoint=public_endpoint,
        )
    )
    resp = client.list_connections(
        request=ListConnectionsRequest(
            connection_name=connection_name, workspace_id=workspace_id, max_results=50
        )
    )
    connection_info = next(
        (
            conn
            for conn in resp.body.connections
            if conn.connection_name == connection_name
        ),
        None,
    )
    if not connection_info:
        raise ValueError(f"Connection {connection_name} not found")
    ls_connection = client.get_connection(
        connection_id=connection_info.connection_id,
        request=GetConnectionRequest(
            workspace_id=workspace_id,
            encrypt_option="PlainText",
        ),
    )
    conn_info = ls_connection.body
    configs = conn_info.configs or {}
    secrets = conn_info.secrets or {}

    logger.info(f"Configs conn_info:\n {conn_info}")
    return conn_info, configs, secrets
