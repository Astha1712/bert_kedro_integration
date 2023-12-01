"""
This is a boilerplate pipeline 'bert'
generated using Kedro 0.18.14
"""
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

from kedro.pipeline import node, Pipeline
from .nodes import bert_language_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                bert_language_model,
                inputs="input_text",
                outputs="bert_predictions",
                name="bert_node",
            ),
            # Add other nodes as needed
        ]
    )
