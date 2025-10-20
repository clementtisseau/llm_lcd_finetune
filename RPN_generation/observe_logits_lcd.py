import torch
import math
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter

import constraintlm as clm
import outlines
from outlines.processors import RegexLogitsProcessor

from pathlib import Path



# --- Utility Functions for File I/O ---
def stream_jsonl(filename: str):
    """
    Parses a JSONL file and yields each line as a dictionary.
    """
    with open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)
# ----- Helper functions -----
import os, re, time, json, shutil, signal
from typing import Optional

def load_training_state(ckpt_dir: Path):
    path = ckpt_dir / "training_state.pt"
    return torch.load(path, map_location="cpu") if path.exists() else {}
# ----- End of Helper functions -----


def observe_logits(
    model_name: str,
    dataset_name: str,
    dataset_size: int = None,
    batch_size: int = 32,
    device: str = None,
):
    # Setup device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading dataset")
    dataset = list(stream_jsonl(dataset_name))
    if dataset_size is not None: dataset = dataset[:dataset_size]
    print("Dataset loaded")

    max_memory3 = {
        0: "7GB",
        1: "7GB",
        2: "7GB"
    }

    model_to_load = model_name
    # Load tokenizer and model
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"Loading CLM model: {model_name}...")
    clm_model = clm.TransformersLM(
        model_to_load,
        torch_dtype=dtype,
        device_map="balanced",
        max_memory=max_memory3,
    )
    tokenizer = clm_model.tokenizer
    tokenizer.padding_side = "left"  # simpler for generating multiple sequences
    model = clm_model.model
    model.eval()
    print("CLM model loaded.")
    print("Creating Logits Processor...")
    outlines_model = outlines.from_transformers(clm_model.model, clm_model.tokenizer)
    rpn_logits_processor = RegexLogitsProcessor(
        r"(?:\d+|[+\-*/])(?: (?:\d+|[+\-*/]))*",
        outlines_model.tokenizer,
        outlines_model.tensor_library_name,
    )
    print("Logits Processor created.")
    model.config.use_cache = False          # Disable KV-cache, which is useless during training

    first_device = next(model.parameters()).device

    
    SYSTEM = (
        "You are an expert at converting arithmetic expressions into Reverse Polish "
        "Notation (RPN). You always output only the RPN expression, with tokens "
        "separated by a single space. Do not include explanations or extra text."
    )
    def build_messages(infix: str, rpn: str | None):
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Infix: {infix}\nRPN:"},
        ]
        if rpn is not None:
            msgs.append({"role": "assistant", "content": rpn})
        return msgs
    
    rows = dataset
    # Prompt-only strings (stop at assistant start) — used to compute prompt lengths
    prompt_texts = [
        tokenizer.apply_chat_template(
            build_messages(r["infix"], None),
            tokenize=False,
            add_generation_prompt=True,   # ends right after assistant-start
            enable_thinking=False
        )
        for r in rows
    ]# example: (4 additional newlines when enable_thinking=False)
    # "<|im_start|>system\nYou are an expert at converting arithmetic expressions into Reverse Polish Notation (RPN). You always output only the RPN expression, with tokens separated by a single space. Do not include explanations or extra text.<|im_end|>\n<|im_start|>user\nInfix: 3 + 4 * 2\nRPN:<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"

    # Full training strings (prompt + target)
    full_texts = [
        tokenizer.apply_chat_template(
            build_messages(r["infix"], r["rpn"]),
            tokenize=False,
            add_generation_prompt=False,  # includes assistant message + proper end token
            enable_thinking=False
        )
        for r in rows
    ]
    # "<|im_start|>system\nYou are an expert at converting arithmetic expressions into Reverse Polish Notation (RPN). You always output only the RPN expression, with tokens separated by a single space. Do not include explanations or extra text.<|im_end|>\n<|im_start|>user\nInfix: 3 + 4 * 2\nRPN:<|im_end|>\n<|im_start|>assistant\n\n\n\n\n3 4 2 * +<|im_end|>\n"


    # Tokenize full sequences
    encodings = tokenizer(
        full_texts,
        add_special_tokens=False,        # template already added them
        padding=True,                    # left-padding
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = input_ids.clone()
    # Compute per-row full (non-pad) lengths
    full_lengths = attention_mask.sum(dim=1)
    padded_seq_len = input_ids.size(1)
    # Prompt lengths (tokenized without padding)
    prompt_tokenized = tokenizer(
        prompt_texts, add_special_tokens=False, padding=False, return_tensors=None
    )
    prompt_lengths = [len(ids) for ids in prompt_tokenized["input_ids"]]
    # Mask the prompt region (left-padded batches)
    for i, l_prompt in enumerate(prompt_lengths):
        start_prompt = padded_seq_len - full_lengths[i]     # index where non-pad starts
        end_prompt = min(start_prompt + l_prompt, padded_seq_len)
        labels[i, start_prompt:end_prompt] = -100
    # Mask padding
    labels[attention_mask == 0] = -100
    # --- END Process dataset ---
  

    # DataLoader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)    # We need shuffle = False if we want to track the input of each sentence
                                                                                    # By default drop_last=False, meaning the last batch is smaller than the other ones

    model.eval()
    
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch 1", miniters=10)):
        batch_input_ids, batch_attention, batch_labels = [x.to(first_device) for x in batch]

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention,
            return_dict=True,
        )
        logits = outputs.logits  # shape (batch, seq_len, vocab_size)  


        biased_logits = logits.clone()
        micro_bs = logits.shape[0]
        for i in range(micro_bs):
            # idx = step * micro_batch_size + i
            # start_prompt = padded_seq_len - full_lengths[idx]
            # len_prompt = prompt_lengths[idx]                                
            # start_ans = min(start_prompt + len_prompt, padded_seq_len)      # except error: start_prompt + len_prompt < padded_seq_len
            # end_ans = padded_seq_len - 2                                    # - 2 : a_1, ..., a_m, <|im_end|>, \n (apply_chat_template add a eos_token_id and a newline)
            valid = (batch_labels[i] != -100).nonzero(as_tuple=False).squeeze(-1)
            start_ans = int(valid[0])                                       # first non -100 label
            end_ans   = int(valid[-1]) - 1                                  # This token is excluded from biasing (-1: because here we have a_1, ..., a_m, <|im_end|>, \n)
            print(f"---input---'{tokenizer.decode(batch_input_ids[i, :], skip_special_tokens=False, clean_up_tokenization_spaces=False)}'---end of input---")
            print(tokenizer.decode(batch_input_ids[i, start_ans-1], skip_special_tokens=False, clean_up_tokenization_spaces=False), "first token to compute and bias logits from")
            print(tokenizer.decode(batch_input_ids[i, end_ans-1], skip_special_tokens=False, clean_up_tokenization_spaces=False), "last token to compute and bias logits from")
            rpn_logits_processor._seq_start_idx = start_ans                 # Length of the prompt before the generation biased of the answer
            rpn_logits_processor._guide_states = {hash(tuple()): rpn_logits_processor.guide.initial_state}        # reset the _guide_states dictionary
            for j in range(start_ans-1, end_ans):
                print(f"generated so far:'{tokenizer.decode(batch_input_ids[i, start_ans-1:j+1])}'", f"token we are predicting from:'{tokenizer.decode(batch_input_ids[i, j])}'")
                biased = rpn_logits_processor.process_logits(batch_input_ids[i, :j+1].unsqueeze(0), logits[i, j, :].unsqueeze(0)).squeeze(0)  # returns shape (batch, seq_len, vocab)
                biased_logits[i, j, :] = biased

                mask = ~torch.isneginf(biased)
                proba = torch.softmax(logits[i, j, :].to(torch.float64), dim=0)
                biased_proba = proba * mask
                biased_proba = biased_proba / biased_proba.sum().clamp_min(torch.finfo(biased_proba.dtype).tiny)

                # print top-10 logits with token ids and decoded tokens
                vals, ids = torch.topk(biased_proba, 10)
                orig_vals = [float(x) for x in proba[ids].detach().cpu().tolist()]  # real (unbiased) probs for same ids
                ids = [int(x) for x in ids.detach().cpu().tolist()]
                vals = [float(x) for x in vals.detach().cpu().tolist()]
                toks = tokenizer.convert_ids_to_tokens(ids) if hasattr(tokenizer, "convert_ids_to_tokens") else [tokenizer.decode([i]) for i in ids]
                print("\n".join(f"{k+1:2d}. proba={u:.12e} | biased_proba={v:.12e} | id={i} | token={t!r}" for k,(i,v,u,t) in enumerate(zip(ids, vals, orig_vals, toks))))


        
if __name__ == "__main__":
    import argparse

    HERE = Path(__file__).resolve().parent  # directory containing assisted_finetune.py
    DATA = HERE / "RPN_benchmark" / "train" / "train_dataset1024.jsonl"     # dataset used for training

    parser = argparse.ArgumentParser(description="Fine-tune a constrained LLM with biased logits.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset_name", type=str, default=DATA)
    parser.add_argument("--dataset_size", type=int, default=4,
                        help="Number of samples of the whole dataset to train on.")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()


    observe_logits(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
    )




# For Qwen3-1.7B (no SFT)
# infix: 7 * 85 + 76 - (41 + 98 - 98 - 33)
# rpn:   7 85 * 76 + 41 98 + 98 - 33 - -
# 
# Here, even thought the model is wrong because it predicts <|im_end|> instead of Ġ-, the model assess a proba of 0.9998 to its error.
# 7 85 * 76 + 41 98 + 98 - 33' token we are predicting from:'3'
#  1. proba=9.999999948411e-01 | biased_proba=9.999999966017e-01 | id=481 | token='Ġ-'
#  2. proba=3.398267801964e-09 | biased_proba=3.398267807947e-09 | id=488 | token='Ġ+'
#  3. proba=3.442477090710e-14 | biased_proba=3.442477096771e-14 | id=608 | token='Ġ/'
#  4. proba=5.279209601103e-15 | biased_proba=5.279209610398e-15 | id=353 | token='Ġ*'
#  5. proba=2.200701976622e-15 | biased_proba=2.200701980497e-15 | id=220 | token='Ġ'
#  6. proba=1.181502516682e-21 | biased_proba=1.181502518762e-21 | id=15 | token='0'
#  7. proba=5.581022708598e-22 | biased_proba=5.581022718424e-22 | id=17 | token='2'
#  8. proba=4.626830334654e-22 | biased_proba=4.626830342800e-22 | id=16 | token='1'
#  9. proba=2.400367956259e-22 | biased_proba=2.400367960485e-22 | id=20 | token='5'
# 10. proba=2.326516398178e-22 | biased_proba=2.326516402274e-22 | id=24 | token='9'
# generated so far:'

# 7 85 * 76 + 41 98 + 98 - 33 -' token we are predicting from:' -'
#  1. proba=9.998067336248e-01 | biased_proba=9.999928026442e-01 | id=151645 | token='<|im_end|>'           # ERROR
#  2. proba=6.143024883678e-06 | biased_proba=6.144168131246e-06 | id=488 | token='Ġ+'
#  3. proba=5.042501891233e-07 | biased_proba=5.043440325983e-07 | id=220 | token='Ġ'
#  4. proba=4.449992300290e-07 | biased_proba=4.450820466050e-07 | id=481 | token='Ġ-'
#  5. proba=9.327690588755e-08 | biased_proba=9.329426518493e-08 | id=608 | token='Ġ/'
#  6. proba=1.046537879794e-08 | biased_proba=1.046732645713e-08 | id=353 | token='Ġ*'
#  7. proba=2.804899579730e-11 | biased_proba=0.000000000000e+00 | id=2 | token='#'
#  8. proba=4.185428469399e-12 | biased_proba=0.000000000000e+00 | id=3 | token='$'
#  9. proba=7.091896760118e-12 | biased_proba=0.000000000000e+00 | id=1 | token='"'
# 10. proba=1.237579733983e-10 | biased_proba=0.000000000000e+00 | id=0 | token='!'
# generated so far:'

# 7 85 * 76 + 41 98 + 98 - 33 - -' token we are predicting from:' -'
#  1. proba=9.999929943350e-01 | biased_proba=9.999996411391e-01 | id=151645 | token='<|im_end|>'
#  2. proba=3.466303128733e-07 | biased_proba=3.466326168733e-07 | id=488 | token='Ġ+'
#  3. proba=1.046732846363e-08 | biased_proba=1.046739803840e-08 | id=220 | token='Ġ'
#  4. proba=1.507961639488e-09 | biased_proba=1.507971662684e-09 | id=481 | token='Ġ-'
#  5. proba=2.385941783683e-10 | biased_proba=2.385957642681e-10 | id=608 | token='Ġ/'
#  6. proba=1.432869414366e-11 | biased_proba=1.432878938435e-11 | id=353 | token='Ġ*'
#  7. proba=8.423404742504e-12 | biased_proba=0.000000000000e+00 | id=2 | token='#'
#  8. proba=4.678484362136e-13 | biased_proba=0.000000000000e+00 | id=3 | token='$'
#  9. proba=1.271743902632e-12 | biased_proba=0.000000000000e+00 | id=1 | token='"'
# 10. proba=4.413548512429e-11 | biased_proba=0.000000000000e+00 | id=0 | token='!'