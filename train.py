from consts import MODEL_ID
from datasets import load_from_disk
import evaluate
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Trainer, TrainingArguments
import wandb

FINETUNED_MODEL_NAME = f"{MODEL_ID.split('/')[-1]}-finetuned-gtzan"
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
gradient_accumulation_steps = 1
NUM_TRAIN_EPOCHS = 10
METRIC = "accuracy"

def get_metrics():
    metric = evaluate.load(METRIC)

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)
    
    return compute_metrics

def get_model():
    id2label_fn = gtzan["train"].features["label"].int2str
    id2label = {
        str(i): id2label_fn(i)
        for i in range(len(gtzan["train"].features["label"].names))
    }
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    return AutoModelForAudioClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

if __name__ == "__main__":
    
    wandb.init(project="hf-audio-u4", job_type="train")

    gtzan = load_from_disk("./data/gtzan")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID, do_normalize=True, return_attention_mask=True
    )

    model = get_model()

    training_args = TrainingArguments(
        FINETUNED_MODEL_NAME,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
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