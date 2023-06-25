import os
import logging
import datasets
import transformers

from pprint import pprint
from itertools import chain
from datasets import load_dataset, concatenate_datasets
from transformers.testing_utils import CaptureLogger
from transformers import AutoTokenizer, LlamaTokenizer


tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

logger = logging.getLogger(__name__)

lm_datasets = []
files = ["data/test_corpus.txt"]
data_cache_dir = "./cache_data"
preprocessing_num_workers = 1

# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
tokenizer = LlamaTokenizer.from_pretrained("ziqingyang/chinese-llama-lora-7b")
tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese")

def print_dict(adict):
  for k,v in adict.items():
    print(k, v)

def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

block_size = 128

# 将所有文本进行拼接
def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

for idx, file in enumerate(files):
    data_file = file
    filename = ''.join(file.split(".")[:-1])

    cache_path = os.path.join(data_cache_dir, filename)
    os.makedirs(cache_path, exist_ok=True)
    try:
        processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
        print(f'training datasets-{filename} has been loaded from disk')
    except Exception:
        cache_dir = os.path.join(data_cache_dir, filename + "_text")
        os.makedirs(cache_dir, exist_ok=True)

        raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
        print_dict(raw_dataset["train"][0])
        # 直接进行tokenize，需要注意的是只需要在句子开头加上bos_token
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns="text",
            load_from_cache_file=True,
            keep_in_memory=False,
            cache_file_names={k: os.path.join(cache_dir, f'tokenized.arrow') for k in raw_dataset},
            desc="Running tokenizer on dataset",
        )

        print_dict(tokenized_dataset["train"][0])

        grouped_datasets = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=True,
            keep_in_memory=False,
            cache_file_names={k: os.path.join(cache_dir, f'grouped.arrow') for k in tokenized_dataset},
            desc=f"Grouping texts in chunks of {block_size}",
        )
        processed_dataset = grouped_datasets

        print_dict(processed_dataset["train"][0])
        processed_dataset.save_to_disk(cache_path)
    if idx == 0:
        lm_datasets = processed_dataset['train']
    else:
        assert lm_datasets.features.type == processed_dataset["train"].features.type
        lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])

lm_datasets = lm_datasets.train_test_split(test_size=0.1)

print_dict(lm_datasets["train"][0])
