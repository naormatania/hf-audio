import argparse
from consts import MODEL_ID
import gradio as gr
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

DATASET_NAME = "marsyas/gtzan"
DATASET_CONFIG = "all"
MAX_DURATION = 30.0
BATCH_SIZE = 100
NUM_PROC = 1
TEST_SIZE = 0.1

def generate_audio(gtzan, id2label_fn):
    example = gtzan["train"].shuffle()[0]
    audio = example["audio"]
    return (
        audio["sampling_rate"],
        audio["array"],
    ), id2label_fn(example["genre"])

def show_demo(gtzan, id2label_fn):
    with gr.Blocks() as demo:
        with gr.Column():
            for _ in range(4):
                audio, label = generate_audio(gtzan, id2label_fn)
                output = gr.Audio(audio, label=label)
    demo.launch(debug=True)

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * MAX_DURATION),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--output_path", type=str)

    args, _ = parser.parse_known_args()

    gtzan = load_dataset(DATASET_NAME, DATASET_CONFIG)
    gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=TEST_SIZE)
    id2label_fn = gtzan["train"].features["genre"].int2str
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID, do_normalize=True, return_attention_mask=True
    )
    gtzan = gtzan.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))

    gtzan_encoded = gtzan.map(
        preprocess_function,
        remove_columns=["audio", "file"],
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
    )
    gtzan_encoded = gtzan_encoded.rename_column("genre", "label")
    gtzan_encoded.save_to_disk(args.output_path)