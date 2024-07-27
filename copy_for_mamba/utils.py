import hashlib
from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from ..extras.logging import get_logger
from collections import defaultdict
if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import TrainingArguments
    from llmtuner.hparams import DataArguments

logger = get_logger(__name__)


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    OBSERVATION = "observation"
    FUNCTION = "function"


def checksum(data_files: List[str], file_sha1: Optional[str] = None) -> None:
    if file_sha1 is None:
        logger.warning("Checksum failed: missing SHA-1 hash value in dataset_info.json.")
        return

    if len(data_files) != 1:
        logger.warning("Checksum failed: too many files.")
        return

    with open(data_files[0], "rb") as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
        if sha1 != file_sha1:
            logger.warning("Checksum failed: mismatched SHA-1 hash value at {}.".format(data_files[0]))


def infer_max_len(source_len: int, target_len: int, data_args: "DataArguments") -> Tuple[int, int]:
    max_target_len = int(data_args.cutoff_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, data_args.reserved_label_len)
    max_source_len = data_args.cutoff_len - max_target_len
    return max_source_len, max_target_len


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    data_args: "DataArguments",
    training_args: "TrainingArguments"
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6: # Split the dataset
            if data_args.streaming:
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                return {"train_dataset": train_set, "eval_dataset": val_set}
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)

                return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            # print(type(dataset))
            # input_ids_first_element = dataset['input_ids'][1:5]
            # attention_mask_first_element = dataset['attention_mask'][1:5]
            # labels_first_element = dataset['labels'][1:5]
            # result_dict = {'input_ids': input_ids_first_element, 'attention_mask': attention_mask_first_element, 'labels': labels_first_element}
            # print(len(result_dict))
            dataset = dataset.train_test_split(train_size=1, seed=training_args.seed)
            # print(dataset['train']['input_ids'])
            # if data_args.streaming:
            #     dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
            # for i in range(len(dataset["train"])):
            #     example = dataset["train"][i]
            #     input_ids = example['input_ids']
            #     labels = example['labels']

            #     print(f"Example {i + 1}:")
            #     print("Input IDs:", input_ids)
            #     print("Labels:", labels)
            #     print("\n")
            # length_counts = {i: 0 for i in range(42, 513)}
            # count = 0
            # cc = 2000
            # sum = 0
            # for input in dataset['input_ids']:
            #     # print(input)
            #     input_length = len(input)
            #     sum += input_length
            #     if len(input) > count:
            #         count = len(input)
            #     if len(input) < cc:
            #         cc = len(input)
            #     length_counts[input_length] += 1
            # length_counts = dict(length_counts)

            # print(f'largest: {count}')
            # print('smallest:', cc)
            # print(f"Number of inputs for each length: {length_counts}")
            # for i in range(42,513):
            #     print(f"{length_counts[i]}")
            # print('avg:', sum/len(dataset['input_ids']))
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
            return {"train_dataset":  dataset['train']}
    else: # do_eval or do_predict
        val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
        dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
        return {"eval_dataset": dataset["test"]}
