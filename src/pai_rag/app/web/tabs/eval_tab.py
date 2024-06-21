import os
from typing import Dict, Any
import gradio as gr
from pai_rag.app.web.rag_client import rag_client
import tempfile
import json
import pandas as pd
from datetime import datetime


def generate_and_download_qa_file(overwrite):
    tmpdir = tempfile.mkdtemp()
    qa_content = rag_client.evaluate_for_generate_qa(bool(overwrite))
    outputPath = os.path.join(tmpdir, "qa_dataset_output.json")
    with open(outputPath, "w", encoding="utf-8") as f:
        json.dump(qa_content, f, ensure_ascii=False, indent=4)

    return outputPath


def eval_retrieval_stage():
    retrieval_res = rag_client.evaluate_for_retrieval_stage()
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pd_results = {
        "Metrics": ["HitRate", "MRR", "LastModified"],
        "Value": [
            retrieval_res["result"]["hit_rate_mean"],
            retrieval_res["result"]["mrr_mean"],
            formatted_time,
        ],
    }
    return pd.DataFrame(pd_results)


def eval_response_stage():
    response_res = rag_client.evaluate_for_response_stage()
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pd_results = {
        "Metrics": ["Faithfulness", "Correctness", "Similarity", "LastModified"],
        "Value": [
            response_res["result"]["faithfulness_mean"],
            response_res["result"]["correctness_mean"],
            response_res["result"]["similarity_mean"],
            formatted_time,
        ],
    }
    return pd.DataFrame(pd_results)


def create_evaluation_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(scale=2):
            generate_btn = gr.Button("Generate QA Dataset", variant="primary")
            overwrite_qa = gr.Checkbox(
                label="Overwrite QA (If the index has changed, regenerate and overwrite the existing QA dataset.)",
                elem_id="overwrite_qa",
                value=False,
            )
            qa_dataset_file = gr.components.File(
                label="\N{rocket} QA Dadaset Results",
                elem_id="qa_dataset_file",
            )
            generate_btn.click(
                fn=generate_and_download_qa_file,
                inputs=overwrite_qa,
                outputs=qa_dataset_file,
            )

        with gr.Column(scale=2):
            eval_retrieval_btn = gr.Button(
                "Evaluate Retrieval Stage", variant="primary"
            )
            eval_retrieval_res = gr.DataFrame(
                label="\N{rocket} Evaluation Resultes for Retrieval ",
                elem_id="eval_retrieval_res",
            )
            eval_retrieval_btn.click(
                fn=eval_retrieval_stage, outputs=eval_retrieval_res
            )

            eval_response_btn = gr.Button("Evaluate Response Stage", variant="primary")
            eval_response_res = gr.DataFrame(
                label="\N{rocket} Evaluation Resultes for Response ",
                elem_id="eval_response_res",
            )
            eval_response_btn.click(fn=eval_response_stage, outputs=eval_response_res)

        return {
            # qa_dataset_file.elem_id: qa_dataset_file,
            eval_retrieval_res.elem_id: eval_retrieval_res,
            eval_response_res.elem_id: eval_response_res,
        }
