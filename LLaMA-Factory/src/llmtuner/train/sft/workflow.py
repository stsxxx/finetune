# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, BitsAndBytesConfig
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from datasets import load_metric
import numpy as np
from datasets import load_dataset

from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model_and_tokenizer
from ...train.sft.metric import ComputeMetrics
from ...train.sft.trainer import CustomSeq2SeqTrainer
from ...train.utils import create_modelcard_and_push
import sys
sys.path.append('/home/stilex/BlackMamba')
from mamba_model import MambaModel

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from ...hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
# metric implementation here
# Some potentially useful helper functions for the metrics
def get_available_device():
    # Check if a GPU is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()

        # Choose the first available GPU (you can modify this logic as needed)
        selected_gpu = torch.device(f"cuda:0")

        print(f"Number of available GPUs: {num_gpus}")
        print(f"Selected GPU: {selected_gpu}")

        return selected_gpu
    else:
        print("No GPU available, using CPU.")
        return torch.device("cpu")
    

def compute_metrics(p):
    predictions = p.predictions 
    predictions = np.argmax(predictions, axis=-1)
    label_ids = p.label_ids 
    print(label_ids.shape)
    if isinstance(predictions, np.ndarray):

        print(predictions.shape)
        predictions = predictions.tolist()
    if isinstance(label_ids, np.ndarray):
        label_ids = label_ids.tolist()
    f1_metric = load_metric("f1")
    exact_match = load_metric("exact_match")
    em_results = exact_match.compute(predictions=predictions, references=label_ids)
    print(predictions[0])
    print(label_ids[0])
    f1_results = f1_metric.compute(predictions=predictions[0], references=label_ids[0], average="weighted")


    # Extract F1 and EM scores
    f1_score = f1_results
    em_score = em_results
    print(em_score)
    print(f1_score)
    return f1_score 

def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):  
    # if training_args.do_train:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    model = MambaModel.from_pretrained(pretrained_model_name="Zyphra/BlackMamba-2.8B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    tokenizer.add_special_tokens({'eos_token': DEFAULT_EOS_TOKEN})
    tokenizer.add_special_tokens({'bos_token': DEFAULT_BOS_TOKEN})
    tokenizer.add_special_tokens({'unk_token': DEFAULT_UNK_TOKEN})
    print(tokenizer.pad_token)
    print(tokenizer.eos_token)
    print(tokenizer.unk_token)
    print(tokenizer.bos_token)


    model = model.to(device=device,dtype = torch.bfloat16)
    # print(model.hf_device_map)
    # print(model._no_split_modules)
    # elif training_args.do_predict:
    #     model_id = "mistralai/Mixtral-8x7B-v0.1"
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     device = get_available_device()
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16
    #     )
    #     model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=bnb_config, device_map=device)
    #     model.load_adapter("/home/stilex/dst/LLaMA-Factory/mixtral/checkpoint-6000")
    #     config_kwargs = {
    #         "trust_remote_code": True,
    #         "cache_dir": model_args.cache_dir,
    #         "revision": model_args.model_revision,
    #         "token": model_args.hf_hub_token
    #     }

    #     tokenizer = AutoTokenizer.from_pretrained(
    #         model_args.model_name_or_path,
    #         use_fast=model_args.use_fast_tokenizer,
    #         split_special_tokens=model_args.split_special_tokens,
    #         padding_side="left",
    #         **config_kwargs
    #     )


    
    dataset = get_dataset(model_args, data_args, tokenizer, training_args, stage="sft")
    print(dataset)
    print(model)
    # print(data_args.val_size)
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True) # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None, # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams,
        save_strategy = 'epoch'
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        # compute_metrics=compute_metrics,
        **split_dataset(dataset, data_args, training_args)
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    gen_kwargs["max_new_tokens"] = 80
    # Training
    epoch = 1
    for i in range(epoch):
        
        if training_args.do_train:

            # with profile(
            # profile_memory=True,
            # with_stack=True,
            # record_shapes=True,
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(
            #     dir_name = "/home/stilex/dst/LLaMA-Factory/finetune_stats", worker_name = "mixtral8x7b_2test")
            # ) as prof:
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            # trainer.save_model()
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
            if trainer.is_world_process_zero() and finetuning_args.plot_loss:
                plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

        # tokenizer.padding_side = "left"

        # tokenizer.eos_token_id = 2
        # tokenizer.bos_token_id = 1
        # val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
        # test_data = load_dataset("json", data_files="/home/stilex/dst/LLaMA-Factory/data/alpaca_data_en_52k.json")
        # print(test_data['train'])
        # test_data = test_data['train'].train_test_split(test_size=val_size, seed=training_args.seed)
        # print(test_data['test'])
        # for example in range(len(test_data['test']['input'])):
        #     print(test_data['test']['input'][example])
        #     print(test_data['test']['instruction'][example])

        # input_encoding = tokenizer(test_data['test']['instruction'],test_data['test']['input'],  truncation=True,return_tensors="pt",padding=True)
        # print(input_encoding)

        # input_string = tokenizer.decode(input_encoding['input_ids'].tolist()[1])
        # print(input_string)
        # # answer_encoding = tokenizer(test_data['test']['output'],  truncation=True,return_tensors="pt")
        # # print(answer_encoding)
        # f1 = []
        # em = []
        # eval_batch = int(training_args.per_device_eval_batch_size)
        # with torch.no_grad():

        #     for i in range(0, len(input_encoding['input_ids']), eval_batch):
        # # Generate the next tokens based on the input batch
        #         output_ids_batch = model.generate(inputs=input_encoding['input_ids'][i:i+eval_batch],attention_mask=input_encoding['attention_mask'][i:i+eval_batch],**gen_kwargs)
        #         for out in range(len(output_ids_batch.tolist())):
        #             print('pos: ',i+out)
        #             output_string = tokenizer.decode(output_ids_batch[out])
        #             answer_string = test_data['test']['output'][i+out]
        #             print(output_string)
        #             print(answer_string)
        #             
        #     # print(f1)
        # f1_score = sum(f1) / len(f1)
        # em_score = sum(em) / len(em)
        # print('f1 socre:', f1_score)
        # print('em score:', em_score)
                        
        # Evaluation
        # if training_args.do_eval:
        #     metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        #     print(metrics)
        #     if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
        #         metrics.pop("eval_loss", None)
        #     trainer.log_metrics("eval", metrics)
        #     trainer.save_metrics("eval", metrics)

        # Predict
        if training_args.do_predict:
            tokenizer.padding_side = "left" # use left-padding in generation
            trainer.tokenizer = tokenizer
            val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
            test_data = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
        if training_args.do_predict:
            predict_results = trainer.predict(test_data['test'], metric_key_prefix="predict", **gen_kwargs)
            if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
                predict_results.metrics.pop("predict_loss", None)
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)
            print(predict_results)
            trainer.save_predictions(predict_results)
            
        # Create model card
        create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
