from typing import Dict, Any
import gradio as gr
from pai_rag.app.web.ui_constants import EMBEDDING_API_KEY_DICT
from pai_rag.app.web.utils import components_to_dict
from pai_rag.app.web.index_utils import index_related_component_keys
from pai_rag.app.web.tabs.vector_db_panel import create_vector_db_panel
import pai_rag.app.web.event_listeners as ev_listeners


def create_setting_tab() -> Dict[str, Any]:
    components = []
    with gr.Row():
        with gr.Column(variant="panel"):
            with gr.Column(scale=5):
                _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **Index**")

                vector_index = gr.Dropdown(
                    label="Index Name",
                    choices=["NEW"],
                    value="NEW",
                    interactive=True,
                    elem_id="vector_index",
                )

                new_index_name = gr.Textbox(
                    label="New Index Name",
                    value="",
                    interactive=True,
                    elem_id="new_index_name",
                    visible=False,
                )

                # _ = gr.Markdown(value="**Index - Embedding Model**")
                embed_source = gr.Radio(
                    EMBEDDING_API_KEY_DICT.keys(),
                    label="Embedding Type",
                    elem_id="embed_source",
                    interactive=True,
                )
                embed_model = gr.Dropdown(
                    label="Embedding Model Name",
                    elem_id="embed_model",
                    visible=False,
                )
                with gr.Row():
                    embed_dim = gr.Textbox(
                        label="Embedding Dimension",
                        elem_id="embed_dim",
                    )
                    embed_batch_size = gr.Textbox(
                        label="Embedding Batch Size",
                        elem_id="embed_batch_size",
                    )
                    embed_type = gr.Textbox(
                        label="Embedding Type",
                        elem_id="embed_type",
                    )
                    embed_api_key = gr.Textbox(
                        label="API KEY",
                        elem_id="embed_api_key",
                        visible=False,
                        type="password",
                    )
            vector_db_elems, vector_db_components = create_vector_db_panel()

            add_index_button = gr.Button(
                "Add Index",
                variant="primary",
                visible=False,
                elem_id="add_index_button",
            )
            update_index_button = gr.Button(
                "Update Index",
                variant="primary",
                visible=False,
                elem_id="update_index_button",
            )
            delete_index_button = gr.Button(
                "Delete Index",
                variant="stop",
                visible=False,
                elem_id="delete_index_button",
            )

            embed_source.input(
                fn=ev_listeners.change_emb_source,
                inputs=[embed_source, embed_model],
                outputs=[embed_model, embed_dim, embed_type, embed_api_key],
            )
            embed_model.input(
                fn=ev_listeners.change_emb_model,
                inputs=[embed_source, embed_model],
                outputs=[embed_dim, embed_type],
            )
            components.extend(
                [
                    embed_source,
                    embed_dim,
                    embed_type,
                    embed_model,
                    embed_api_key,
                    embed_batch_size,
                    vector_index,
                    new_index_name,
                    add_index_button,
                    update_index_button,
                    delete_index_button,
                ]
            )

            all_component = {element.elem_id: element for element in vector_db_elems}
            all_component.update(
                {component.elem_id: component for component in components}
            )
            index_related_components = [
                all_component[key] for key in index_related_component_keys
            ]
            add_index_button.click(
                fn=ev_listeners.add_index,
                inputs=index_related_components,
                outputs=[
                    vector_index,
                    new_index_name,
                    add_index_button,
                    update_index_button,
                    delete_index_button,
                ],
            )

            update_index_button.click(
                fn=ev_listeners.update_index,
                inputs=index_related_components,
                outputs=[
                    vector_index,
                    new_index_name,
                    add_index_button,
                    update_index_button,
                    delete_index_button,
                ],
            )

            """
            delete_index_button.click(
                fn=ev_listeners.delete_index,
                inputs=[vector_index],
                outputs=[],
                visible=False,
            )
            """

        with gr.Column(variant="panel"):
            with gr.Column(variant="panel"):
                _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **Large Language Model**")
                with gr.Row():
                    llm_base_url = gr.Textbox(
                        label="LLM Base URL",
                        elem_id="llm_base_url",
                        interactive=True,
                        placeholder="Open AI compatible url, e.g. https://api.openai.com/v1",
                    )
                    llm_api_key = gr.Textbox(
                        label="API Key",
                        elem_id="llm_api_key",
                        type="password",
                        interactive=True,
                    )
                    llm_model_name = gr.Textbox(
                        label="Model Name",
                        elem_id="llm_model_name",
                        placeholder="Model Name, e.g. qwen-max, gpt-4",
                        interactive=True,
                    )

            with gr.Column(variant="panel"):
                _ = gr.Markdown(
                    value="\N{WHITE MEDIUM STAR} **(Optional) Multi-Modal Large Language Model**"
                )
                use_mllm = gr.Checkbox(
                    label="Use Multi-Modal LLM",
                    elem_id="use_mllm",
                    container=False,
                )
                with gr.Row(visible=False, elem_id="use_mllm_col") as use_mllm_col:
                    mllm_base_url = gr.Textbox(
                        label="Multimodal-LLM Base URL",
                        elem_id="mllm_base_url",
                        interactive=True,
                        placeholder="Open AI compatible url, e.g. https://api.openai.com/v1",
                    )
                    mllm_api_key = gr.Textbox(
                        label="API Key",
                        elem_id="mllm_api_key",
                        type="password",
                        interactive=True,
                    )
                    mllm_model_name = gr.Textbox(
                        label="Multimodal-LLM Model Name",
                        elem_id="mllm_model_name",
                        interactive=True,
                        placeholder="Model Name, e.g. qwen-vl-max",
                    )
            with gr.Column(scale=5, variant="panel"):
                _ = gr.Markdown(
                    value="\N{WHITE MEDIUM STAR} **(Optional, for saving image & load data) OSS Bucket**"
                )
                use_oss = gr.Checkbox(
                    label="Use OSS Storage",
                    elem_id="use_oss",
                    container=False,
                )
                with gr.Row(visible=False, elem_id="use_oss_col") as use_oss_col:
                    oss_bucket = gr.Textbox(
                        label="OSS Bucket",
                        elem_id="oss_bucket",
                    )
                    oss_ak = gr.Textbox(
                        label="Access Key",
                        elem_id="oss_ak",
                        type="password",
                    )
                    oss_sk = gr.Textbox(
                        label="Access Secret",
                        elem_id="oss_sk",
                        type="password",
                    )
                    oss_endpoint = gr.Textbox(
                        label="OSS Endpoint",
                        elem_id="oss_endpoint",
                        placeholder="oss-cn-hangzhou.aliyuncs.com",
                    )
                use_oss.input(
                    fn=ev_listeners.change_use_oss,
                    inputs=use_oss,
                    outputs=use_oss_col,
                )

            llm_components = [
                llm_base_url,
                llm_model_name,
                llm_api_key,
                use_mllm,
                mllm_base_url,
                mllm_model_name,
                mllm_api_key,
                use_oss,
                oss_ak,
                oss_sk,
                oss_endpoint,
                oss_bucket,
            ]

            components.extend(llm_components)

            use_mllm.input(
                fn=ev_listeners.choose_use_mllm,
                inputs=use_mllm,
                outputs=[use_mllm_col],
            )

            save_btn = gr.Button("Save Llm Setting", variant="primary")
            save_state = gr.Textbox(
                label="Connection Info: ", container=False, visible=False
            )
            save_btn.click(
                fn=ev_listeners.save_config,
                inputs=set(llm_components),
                outputs=[oss_ak, oss_sk, save_state],
                api_name="save_config",
            )
    elems = components_to_dict(components)
    elems.update(vector_db_components)
    elems.update(
        {
            use_oss_col.elem_id: use_oss_col,
            use_mllm_col.elem_id: use_mllm_col,
        }
    )
    return elems
