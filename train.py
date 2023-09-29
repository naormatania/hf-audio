import argparse
from consts import MODEL_ID
from datasets import load_from_disk
import evaluate
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Trainer, TrainingArguments
import wandb
import os

BATCH_SIZE = 8
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 1
NUM_TRAIN_EPOCHS = 15
METRIC = "accuracy"

def get_metrics():
    metric = evaluate.load(METRIC)

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)
    
    return compute_metrics

def get_model(gtzan, config, model_name):
    id2label_fn = gtzan["train"].features["label"].int2str
    id2label = {
        str(i): id2label_fn(i)
        for i in range(len(gtzan["train"].features["label"].names))
    }
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    '''
    if VERSION == "v1":
        print("V1 is running")
        config.update({'layerdrop': 0.5})
    elif VERSION == "v2":
        print("V2 is running")
        config.update({'hidden_dropout': 0.3, 'final_dropout': 0.3})
    elif VERSION == "v3":
        print("V2 is running")
        config.update({'hidden_dropout': 0.5, 'final_dropout': 0.5})
    '''

    return AutoModelForAudioClassification.from_pretrained(
        model_name if wandb.run.resumed else MODEL_ID,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        **config,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--layerdrop", type=float, default=0.1)
    parser.add_argument("--hidden_dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--final_dropout", type=float, default=0.1)
    parser.add_argument("--feat_proj_dropout", type=float, default=0.1)

    # Data, model, and output directories
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--version", type=str)
    parser.add_argument("--wandb_run_id", type=str, default="")

    args, _ = parser.parse_known_args()

    if args.wandb_run_id != "":
        os.environ["WANDB_RESUME"] = "must"
        os.environ["WANDB_RUN_ID"] = args.wandb_run_id

    wandb.init(project="hf-audio-u4", job_type="train", name=f"run-{args.version}")

    gtzan = load_from_disk(args.dataset_path)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID, do_normalize=True, return_attention_mask=True
    )

    config = {
        'layerdrop': args.layerdrop,
        'hidden_dropout': args.hidden_dropout,
        'attention_dropout': args.attention_dropout,
        'final_dropout': args.final_dropout,
        'feat_proj_dropout': args.feat_proj_dropout,
    }
    model_name = f"{MODEL_ID.split('/')[-1]}-finetuned-gtzan-{args.version}"
    model = get_model(gtzan, config, model_name)
    print(model.config)

    print(f"wandb.run.step: {wandb.run.step}")
    num_epoch = NUM_TRAIN_EPOCHS - wandb.run.step

    training_args = TrainingArguments(
        model_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        # Number of updates steps to accumulate the gradients for, before performing a backward/update pass
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=num_epoch,
        # Ratio of total training steps used for a linear warmup from 0 to learning_rate
        warmup_ratio=0.1,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=METRIC,
        # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training
        fp16=True,
        push_to_hub=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=gtzan["train"],
        eval_dataset=gtzan["test"],
        tokenizer=feature_extractor,
        compute_metrics=get_metrics(),
    )

    trainer.train()