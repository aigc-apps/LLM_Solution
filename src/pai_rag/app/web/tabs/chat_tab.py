from typing import Dict, Any, List
import gradio as gr
from pai_rag.app.web.rag_client import RagApiError, rag_client
from loguru import logger


def clear_history(chatbot):
    rag_client.clear_history()
    chatbot = []
    return chatbot, 0


def reset_textbox():
    return gr.update(value="")


def change_search_model_argument(search_type):
    return [
        gr.update(visible=True if search_type == "bing" else False),
        gr.update(visible=True),
        gr.update(visible=True if search_type == "bing" else False),
        gr.update(visible=False if search_type == "bing" else True),
        gr.update(visible=False if search_type == "bing" else True),
        gr.update(visible=False if search_type == "bing" else True),
    ]


def respond(input_elements: List[Any]):
    update_dict = {}

    for element, value in input_elements.items():
        update_dict[element.elem_id] = value

    # empty input.
    if not update_dict["question"]:
        yield update_dict["chatbot"]
        return

    try:
        rag_client.patch_config(update_dict)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    chatbot = update_dict["chatbot"]
    if not update_dict["include_history"]:
        chatbot, _ = clear_history(chatbot)

    query_type = update_dict["query_type"]
    question = update_dict["question"]
    q_msg = {"content": question, "role": "user"}
    chatbot.append(q_msg)
    is_streaming = update_dict["is_streaming"]
    index_name = update_dict["chat_index"]
    citation = update_dict["citation"]

    if chatbot is not None:
        chatbot.append(
            {"content": "", "role": "assistant", "metadata": {"status": "pending"}}
        )
        yield chatbot

    try:
        if query_type == "LLM":
            response_gen = rag_client.query_llm(
                question,
                with_history=update_dict["include_history"],
                stream=is_streaming,
            )
        elif query_type == "Retrieval":
            response_gen = rag_client.query_vector(question, index_name=index_name)

        elif query_type == "RAG (Search Web)":
            response_gen = rag_client.query_search(
                question,
                with_history=update_dict["include_history"],
                stream=is_streaming,
                citation=citation,
            )
        else:
            response_gen = rag_client.query(
                question,
                with_history=update_dict["include_history"],
                stream=is_streaming,
                citation=citation,
                index_name=index_name,
            )

        is_thinking = False
        for resp in response_gen:
            if resp.delta == "<think>":
                chatbot[-1]["metadata"]["title"] = "thinking..."
                chatbot[-1]["metadata"]["log"] = ""
                is_thinking = True

            elif resp.delta == "</think>":
                chatbot[-1]["metadata"]["title"] = "thought"
                chatbot[-1]["metadata"]["status"] = "done"
                is_thinking = False
                chatbot.append(
                    {
                        "content": "",
                        "role": "assistant",
                        "metadata": {"status": "pending"},
                    }
                )

            else:
                if is_thinking:
                    chatbot[-1]["metadata"]["log"] += resp.delta
                else:
                    chatbot[-1]["content"] += resp.delta
            yield chatbot

    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")
    except Exception as e:
        raise gr.Error(f"Error: {e}")
    finally:
        logger.info(f"Chatbot finished: {chatbot}")
        yield chatbot


def create_chat_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(scale=2):
            chat_index = gr.Dropdown(
                choices=[],
                value="",
                label="\N{bookmark} Index Name",
                elem_id="chat_index",
            )
            query_type = gr.Radio(
                ["Retrieval", "LLM", "RAG (Search Web)", "RAG (Retrieval + LLM)"],
                label="\N{fire} Which query do you want to use?",
                elem_id="query_type",
                value="RAG (Retrieval + LLM)",
            )
            is_streaming = gr.Checkbox(
                label="Streaming Output",
                info="Streaming Output",
                elem_id="is_streaming",
                value=True,
            )
            citation = gr.Checkbox(
                label="Citation",
                info="Need Citation",
                elem_id="citation",
                value=False,
            )
            need_image = gr.Checkbox(
                label="Display Image",
                info="Inference with multi-modal LLM.",
                elem_id="need_image",
            )

            with gr.Column(visible=True) as vs_col:
                vec_model_argument = gr.Accordion(
                    "Parameters of Vector Retrieval", open=False
                )
                with vec_model_argument:
                    retrieval_mode = gr.Radio(
                        ["Embedding Only", "Keyword Only", "Hybrid"],
                        label="Retrieval Mode",
                        elem_id="retrieval_mode",
                    )

                    vector_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.7,
                        elem_id="vector_weight",
                        label="Weight of embedding retrieval results",
                        visible=(retrieval_mode == "Hybrid"),
                    )
                    keyword_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=float(1 - vector_weight.value),
                        elem_id="keyword_weight",
                        label="Weight of keyword retrieval results",
                        interactive=False,
                        visible=(retrieval_mode == "Hybrid"),
                    )

                    similarity_top_k = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        elem_id="similarity_top_k",
                        label="Text Top K (choose between 0 and 100)",
                    )
                    image_similarity_top_k = gr.Slider(
                        minimum=0,
                        maximum=10,
                        step=1,
                        elem_id="image_similarity_top_k",
                        label="Image Top K (choose between 0 and 10)",
                    )
                    similarity_threshold = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        elem_id="similarity_threshold",
                        label="Similarity Score Threshold (The more similar the items, the bigger the value.)",
                    )

                    reranker_type = gr.Radio(
                        ["no-reranker", "model-based-reranker"],
                        label="Reranker Type",
                        elem_id="reranker_type",
                    )
                    with gr.Column(
                        visible=(reranker_type == "model-based-reranker"),
                        elem_id="model_reranker_col",
                    ) as model_reranker_col:
                        reranker_model = gr.Radio(
                            [
                                "bge-reranker-base",
                                "bge-reranker-large",
                            ],
                            label="Re-Ranker Model (Note: It will take a long time to load the model when using it for the first time.)",
                            elem_id="reranker_model",
                        )
                        reranker_similarity_threshold = gr.Slider(
                            minimum=-10,
                            maximum=10,
                            step=0.01,
                            elem_id="reranker_similarity_threshold",
                            label="Reranker Similarity Score Threshold (The more similar the items, the bigger the value.)",
                        )
                        reranker_similarity_top_k = gr.Slider(
                            minimum=0,
                            maximum=50,
                            step=1,
                            elem_id="reranker_similarity_top_k",
                            label="Reranker Text Top K (choose between 0 and 50)",
                        )

                    def change_weight(change_weight):
                        return round(float(1 - change_weight), 2)

                    vector_weight.input(
                        fn=change_weight,
                        inputs=vector_weight,
                        outputs=[keyword_weight],
                    )

                    def change_reranker_type(reranker_type):
                        if reranker_type == "no-reranker":
                            return {
                                model_reranker_col: gr.update(visible=False),
                            }
                        elif reranker_type == "model-based-reranker":
                            return {
                                model_reranker_col: gr.update(visible=True),
                            }
                        else:
                            return {
                                model_reranker_col: gr.update(visible=False),
                            }

                    def change_retrieval_mode(retrieval_mode):
                        if retrieval_mode == "Hybrid":
                            return {
                                vector_weight: gr.update(visible=True),
                                keyword_weight: gr.update(visible=True),
                            }
                        else:
                            return {
                                vector_weight: gr.update(visible=False),
                                keyword_weight: gr.update(visible=False),
                            }

                    reranker_type.input(
                        fn=change_reranker_type,
                        inputs=reranker_type,
                        outputs=[model_reranker_col],
                    )

                    retrieval_mode.input(
                        fn=change_retrieval_mode,
                        inputs=retrieval_mode,
                        outputs=[vector_weight, keyword_weight],
                    )

                vec_args = {
                    retrieval_mode,
                    reranker_type,
                    vector_weight,
                    keyword_weight,
                    similarity_top_k,
                    image_similarity_top_k,
                    similarity_threshold,
                    reranker_similarity_threshold,
                    reranker_model,
                    reranker_similarity_top_k,
                }

            with gr.Column(visible=True) as llm_col:
                model_argument = gr.Accordion("Inference Parameters of LLM", open=False)
                with model_argument:
                    llm_temperature = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.001,
                        value=0.1,
                        elem_id="llm_temperature",
                        label="Temperature (choose between 0 and 1)",
                    )
                llm_args = {llm_temperature}

            with gr.Column(visible=False) as search_col:
                search_model_argument = gr.Accordion(
                    "Parameters of Web Search", open=False
                )
                with search_model_argument:
                    search_type = gr.Radio(
                        ["bing", "夸克"],
                        label="Search Engine",
                        elem_id="search_type",
                    )
                    search_api_key = gr.Text(
                        label="Bing API Key",
                        value="",
                        type="password",
                        elem_id="search_api_key",
                    )
                    search_count = gr.Slider(
                        label="Search Count",
                        minimum=5,
                        maximum=50,
                        step=1,
                        elem_id="search_count",
                    )
                    search_lang = gr.Radio(
                        label="Language",
                        choices=["zh-CN", "en-US"],
                        value="zh-CN",
                        elem_id="search_lang",
                    )
                    quark_host = gr.Text(
                        label="Quark Host",
                        value="",
                        elem_id="quark_host",
                    )
                    quark_user = gr.Text(
                        label="Quark User",
                        value="",
                        elem_id="quark_user",
                    )
                    quark_secret = gr.Text(
                        label="Quark Secret",
                        value="",
                        type="password",
                        elem_id="quark_secret",
                    )
                search_args = {
                    search_type,
                    search_api_key,
                    search_count,
                    search_lang,
                    quark_host,
                    quark_user,
                    quark_secret,
                }
                search_type.input(
                    fn=change_search_model_argument,
                    inputs=[search_type],
                    outputs=[
                        search_api_key,
                        search_count,
                        search_lang,
                        quark_host,
                        quark_user,
                        quark_secret,
                    ],
                )

            with gr.Column(visible=True) as lc_col:
                with gr.Tab("Prompt"):
                    text_qa_template = gr.Textbox(
                        label="Prompt Template",
                        value="",
                        elem_id="text_qa_template",
                        lines=10,
                        interactive=True,
                    )
                    citation_text_qa_template = gr.Textbox(
                        label="Citation Prompt Template",
                        value="",
                        elem_id="citation_text_qa_template",
                        lines=10,
                        interactive=True,
                    )
                with gr.Tab("MultiModal Prompt"):
                    multimodal_qa_template = gr.Textbox(
                        label="Multi-modal Prompt Template",
                        value="",
                        elem_id="multimodal_qa_template",
                        lines=12,
                        interactive=True,
                    )
                    citation_multimodal_qa_template = gr.Textbox(
                        label="Citation Multi-modal Prompt Template",
                        value="",
                        elem_id="citation_multimodal_qa_template",
                        lines=12,
                        interactive=True,
                    )

            cur_tokens = gr.Textbox(
                label="\N{fire} Current total count of tokens", visible=False
            )

            def change_query_radio(query_type):
                if query_type == "Retrieval":
                    return {
                        vs_col: gr.update(visible=True),
                        vec_model_argument: gr.update(open=True),
                        search_model_argument: gr.update(open=False),
                        search_col: gr.update(visible=False),
                        llm_col: gr.update(visible=False),
                        model_argument: gr.update(open=False),
                        lc_col: gr.update(visible=False),
                    }
                elif query_type == "LLM":
                    return {
                        vs_col: gr.update(visible=False),
                        vec_model_argument: gr.update(open=False),
                        search_model_argument: gr.update(open=False),
                        search_col: gr.update(visible=False),
                        llm_col: gr.update(visible=True),
                        model_argument: gr.update(open=True),
                        lc_col: gr.update(visible=False),
                    }
                elif query_type == "RAG (Retrieval + LLM)":
                    return {
                        vs_col: gr.update(visible=True),
                        vec_model_argument: gr.update(open=False),
                        search_model_argument: gr.update(open=False),
                        search_col: gr.update(visible=False),
                        llm_col: gr.update(visible=True),
                        model_argument: gr.update(open=False),
                        lc_col: gr.update(visible=True),
                    }
                elif query_type == "RAG (Search Web)":
                    return {
                        vs_col: gr.update(visible=False),
                        vec_model_argument: gr.update(open=False),
                        search_model_argument: gr.update(open=True),
                        search_col: gr.update(visible=True),
                        llm_col: gr.update(visible=False),
                        model_argument: gr.update(open=False),
                        lc_col: gr.update(visible=False),
                    }

            query_type.input(
                fn=change_query_radio,
                inputs=query_type,
                outputs=[
                    vs_col,
                    vec_model_argument,
                    search_model_argument,
                    search_col,
                    llm_col,
                    model_argument,
                    lc_col,
                ],
            )

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(height=500, elem_id="chatbot", type="messages")
            with gr.Row():
                include_history = gr.Checkbox(
                    label="Chat history",
                    info="Query with chat history.",
                    elem_id="include_history",
                    value=False,
                    scale=1,
                )
                question = gr.Textbox(
                    label="Enter your question.", elem_id="question", scale=9
                )
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                clearBtn = gr.Button("Clear History", variant="secondary")

        chat_args = (
            {
                text_qa_template,
                multimodal_qa_template,
                citation_text_qa_template,
                citation_multimodal_qa_template,
                question,
                query_type,
                chatbot,
                is_streaming,
                citation,
                need_image,
                include_history,
                chat_index,
            }
            .union(vec_args)
            .union(llm_args)
            .union(search_args)
        )

        submitBtn.click(
            respond,
            chat_args,
            [chatbot],
            api_name="respond_clk",
        )
        question.submit(
            respond,
            chat_args,
            [chatbot],
            api_name="respond_q",
        )
        submitBtn.click(
            reset_textbox,
            [],
            [question],
            api_name="reset_clk",
        )
        question.submit(
            reset_textbox,
            [],
            [question],
            api_name="reset_q",
        )

        clearBtn.click(clear_history, [chatbot], [chatbot, cur_tokens])
        return {
            chat_index.elem_id: chat_index,
            similarity_top_k.elem_id: similarity_top_k,
            image_similarity_top_k.elem_id: image_similarity_top_k,
            need_image.elem_id: need_image,
            retrieval_mode.elem_id: retrieval_mode,
            reranker_type.elem_id: reranker_type,
            reranker_model.elem_id: reranker_model,
            vector_weight.elem_id: vector_weight,
            keyword_weight.elem_id: keyword_weight,
            similarity_threshold.elem_id: similarity_threshold,
            reranker_similarity_threshold.elem_id: reranker_similarity_threshold,
            reranker_similarity_top_k.elem_id: reranker_similarity_top_k,
            multimodal_qa_template.elem_id: multimodal_qa_template,
            citation_multimodal_qa_template.elem_id: citation_multimodal_qa_template,
            citation_text_qa_template.elem_id: citation_text_qa_template,
            text_qa_template.elem_id: text_qa_template,
            search_lang.elem_id: search_lang,
            search_api_key.elem_id: search_api_key,
            search_count.elem_id: search_count,
            search_type.elem_id: search_type,
            quark_host.elem_id: quark_host,
            quark_secret.elem_id: quark_secret,
            quark_user.elem_id: quark_user,
            model_reranker_col.elem_id: model_reranker_col,
            llm_temperature.elem_id: llm_temperature,
        }
