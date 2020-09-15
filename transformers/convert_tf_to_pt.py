# Original name: convert_bert_original_tf_checkpoint_to_pytorch.py
# https://github.com/huggingface/transformers/blob/master/src/transformers/convert_bert_original_tf_checkpoint_to_pytorch.py


# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BERT checkpoint."""


import argparse

import torch

from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    language = 'danish'

    parser.add_argument(
        "--tf_checkpoint_path",
        default="./"+language+"_bert_uncased_local/bert_model.ckpt",
        type=str,
        required=False,
        help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--bert_config_file",
        default="./"+language+"_bert_uncased_local/bert_config.json",
        type=str,
        required=False,
        help="The config json file corresponding to the pre-trained BERT model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default="./output_"+language+"_local/pytorch_model.bin",
        type=str,
        required=False,
        help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                    args.bert_config_file,
                                    args.pytorch_dump_path)
