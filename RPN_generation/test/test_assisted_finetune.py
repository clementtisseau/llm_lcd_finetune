import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

import constraintlm as clm
import outlines
from outlines.processors import RegexLogitsProcessor

from pathlib import Path

import json
import re

HERE = Path(__file__).resolve().parent  # test
DATA = HERE.parent / "RPN_benchmark" / "data" / "dataset.jsonl"


# --- Utility Functions for File I/O ---
def stream_jsonl(filename: str):
    """
    Parses a JSONL file and yields each line as a dictionary.
    """
    with open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)



def test_finetune(
    model_name: str,
    dataset_name: str,
    dataset_size: int,
    max_length: int = 512,
    batch_size: int = 64,
    device: str = None,
):
    # Setup device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading dataset")
    dataset = list(stream_jsonl(DATA))
    # problems = problems[:10]
    print("Dataset loaded")

    max_memory3 = {
        0: "10GB",
        1: "10GB",
        2: "10GB"
    }
    max_memory2 = {
        0: "75GB",   
        1: "75GB",
    }

    # Load tokenizer and model
    print(f"Loading CLM model: {model_name}...")
    clm_model = clm.TransformersLM(
        model_name,
        torch_dtype=torch.float16,
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
    # １ ２ ０ ３ ５ ４ ６ ８ ７ ９ are fullwidth digits used in East Asian typography. 
    # outlines.processors.RegexLogitsProcessor compiles the regex into a byte-level automaton and (by design) treats \d as ASCII digits only (0–9). 
    # -> it blocks fullwidth digits.
    # 
    print("Logits Processor created.")

    first_device = next(model.parameters()).device

    # # --- Process dataset ---
    # SYSTEM = (
    #     "You are an expert at converting arithmetic expressions into Reverse Polish "
    #     "Notation (RPN). You always output only the RPN expression, with tokens "
    #     "separated by a single space. Do not include explanations or extra text."
    # )
    # def render_prompt(infix: str) -> str:
    #     # prompt only (no target), ends right after assistant-start
    #     return (
    #         "<|im_start|>system\n"
    #         f"{SYSTEM}\n"
    #         "<|im_end|>\n"
    #         "<|im_start|>user\n"
    #         f"Infix: {infix}\n"
    #         "RPN:\n"
    #         "<|im_end|>\n"
    #         "<|im_start|>assistant\n"
    #     )
    # def render_full(infix: str, rpn: str) -> str:
    #     # full training sample = prompt + answer + end
    #     return render_prompt(infix) + f"{rpn}<|im_end|>"
    # # dataset is a LIST of dicts like {"infix": "...", "rpn": "..."}
    # rows = dataset  # keep your current variable name if needed
    # # Build texts for SFT
    # texts = [render_full(r["infix"], r["rpn"]) for r in rows]
    # # Tokenize
    # encodings = tokenizer(
    #     texts,
    #     add_special_tokens=False,           # No need to add_special_tokens, we are adding them by hand
    #     padding=True,
    #     return_tensors="pt",
    # )
    # # Create input_ids, attention_mask, labels
    # input_ids = encodings["input_ids"]
    # padded_seq_len = input_ids.size(1)
    # attention_mask = encodings["attention_mask"]
    # full_lengths = attention_mask.sum(dim=1)            # number of non-pad tokens per row
    # labels = input_ids.clone()
    # prompts = [tokenizer(render_prompt(r["infix"]), add_special_tokens=False)["input_ids"] for r in rows]               # No need to add_special_tokens, we are adding them by hand
    # prompt_lengths = [len(tokenizer(render_prompt(r["infix"]), add_special_tokens=False)["input_ids"]) for r in rows]   # No need to add_special_tokens, we are adding them by hand
    # # mask the prompt region, correctly offset for left padding
    # for i, l_prompt in enumerate(prompt_lengths):
    #     start_prompt = padded_seq_len - full_lengths[i]         
    #     end_prompt = min(start_prompt + l_prompt, padded_seq_len)   # except error: start_prompt + l_prompt < padded_seq_len
    #     labels[i, start_prompt:end_prompt] = -100                   # start_prompt:end_prompt excludes <\s>: <s>, prompt_1, ..., prompt_n, <\s>
    # # mask padding
    # labels[attention_mask == 0] = -100
    # # --- END OF Process dataset ---


    # --- Process dataset ---
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
    ]

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

    # --- Regex helpers ---
    # We consider a next-token candidate "valid" if the decoded answer (from start_ans onward)
    # is either a full match of the RPN regex OR a valid prefix that ends with a single space,
    # meaning the model can still place the next token to complete another item.
    _RPN_FULL = re.compile(r'(?a)(?:\d+|[+\-*/])(?: (?:\d+|[+\-*/]))*')         # (?a) is an inline flag that tells Python’s re engine to use ASCII-only matching for certain character classes
    _RPN_PREFIX = re.compile(r'(?a)(?:\d+|[+\-*/])(?: (?:\d+|[+\-*/]))* ')      # We don't want to accept １ ... ９ fullwidth digits (used in East Asian typography). 
    _eos_txt = None
    if getattr(tokenizer, "eos_token_id", None) is not None:
        _eos_txt = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False) or None
    def _clean_specials(s: str) -> str:
        if _eos_txt:
            return s.replace(_eos_txt, "")
        return s
    # --- END OF Regex helpers ---


    # Arbitrary micro_batch_size
    micro_batch_size = 4

    # DataLoader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=False)    # We need shuffle = False if we want to track the input of each sentence
                                                                                    # By default drop_last=False, meaning the last batch is smaller than the other ones


    model.train()

    for step, batch in enumerate(tqdm(dataloader, desc=f"Test", miniters=10)):
        
        batch_input_ids, batch_attention, batch_labels = [x.to(first_device) for x in batch]

        n_batch, seq_len_batch = batch_input_ids.shape      # (batch, seq_len)
        V = tokenizer.vocab_size
        V_special = len(tokenizer)
        logits_size = model.lm_head.out_features       
        
        logits = torch.zeros(n_batch, seq_len_batch, logits_size)     # shape (batch, seq_len, logits_size)  
        logits[..., V_special:] = float("-inf")                     # we've created dummy logits to see if the biasing of the logits is correct


        biased_logits = logits.clone()
        micro_bs = logits.shape[0]
        for i in range(micro_bs):
            idx = step * micro_batch_size + i
            start_prompt = padded_seq_len - full_lengths[idx]
            len_prompt = prompt_lengths[idx] 
            start_ans = min(start_prompt + len_prompt, padded_seq_len)      # except error: start_prompt + len_prompt < padded_seq_len
            end_ans = padded_seq_len - 2                                    # - 2 : a_1, ..., a_m, <|im_end|>, \n (apply_chat_template add a eos_token_id and a newline)
            print(f"---input---'{tokenizer.decode(batch_input_ids[i, :], skip_special_tokens=False, clean_up_tokenization_spaces=False)}'---end of input---")
            # print(tokenizer.decode(prompts[idx], skip_special_tokens=False, clean_up_tokenization_spaces=False), "this was the prompt")
            # print(tokenizer.decode(batch_input_ids[i, start_ans-1], skip_special_tokens=False, clean_up_tokenization_spaces=False), "first token to compute logits from")    
            rpn_logits_processor._seq_start_idx = start_ans                 # Length of the prompt before the generation biased of the answer
            rpn_logits_processor._guide_states = {hash(tuple()): rpn_logits_processor.guide.initial_state}        # reset the _guide_states dictionary
            for j in range(start_ans-1, end_ans):
                print(f"generated so far:'{tokenizer.decode(batch_input_ids[i, start_ans-1:j+1])}'", f"token we are predicting from:'{tokenizer.decode(batch_input_ids[i, j])}'")
                biased = rpn_logits_processor.process_logits(batch_input_ids[i, :j+1].unsqueeze(0), logits[i, j, :].unsqueeze(0)).squeeze(0)  # returns shape (batch, seq_len, vocab)
                biased_logits[i, j, :] = biased
                # Check if the biased_logits are "correct"
                # The regex is r"(?:\d+|[+\-*/])(?: (?:\d+|[+\-*/]))*"
                # For each candidate v with original logits != -inf, build the text of
                # "answer so far + candidate token" (decoded from start_ans) and verify:
                # - If that decoded string is a valid full match OR a valid prefix (trailing space),
                #   then biased[v] must be allowed (!= -inf).
                # - Otherwise, biased[v] must be disallowed (== -inf).
                with torch.no_grad():
                    orig_allowed_mask = logits[i, j, :].ne(float("-inf"))   # Allowed tokens
                    biased_allowed_mask = biased.ne(float("-inf"))          # Allowed tokens after masking 

                    answer_prefix_ids = batch_input_ids[i, start_ans:j+1]   # not start_ans - 1 because here we consider the generated tokens

                    mismatches = []
                    candidate_ids = torch.nonzero(orig_allowed_mask, as_tuple=False).squeeze(1) # Finds all indices where the value is True
                    
                    for v in candidate_ids.tolist():    # we consider all valid tokens here (not the non-biased ones only)
                        candidate_seq = torch.cat([answer_prefix_ids, torch.tensor([v], device=answer_prefix_ids.device)], dim=0)
                        decoded_candidate = tokenizer.decode(
                            candidate_seq,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False
                        ) 
                        if v == tokenizer.eos_token_id:
                            decoded_candidate = _clean_specials(decoded_candidate)
                            regex_ok = bool(_RPN_FULL.fullmatch(decoded_candidate))   # does it respect the regex (full)?
                        else:
                            regex_ok = bool(_RPN_FULL.fullmatch(decoded_candidate)) or bool(_RPN_PREFIX.fullmatch(decoded_candidate))   # does it respect the regex (full or prefix)?

                        allowed_by_bias = bool(biased_allowed_mask[v].item())       # did the processor allow/ban it?

                        # # --- DEBUGGING FOR EOS ---
                        # It shows if eos_token is authorized when it should (when the generated text could match the regex)
                        # if v == tokenizer.eos_token_id and allowed_by_bias:  
                        #     current_decoded = tokenizer.decode(
                        #             answer_prefix_ids,
                        #             skip_special_tokens=False,
                        #             clean_up_tokenization_spaces=False,
                        #         )
                        #     print(
                        #         "\n [EOS allowed by LogitsProcessor]",
                        #         f"\n  Decoded answer so far: '{current_decoded}'",
                        #         f"\n  Decoded candidate (with EOS): '{decoded_candidate}'",
                        #         f"\n  Regex full match? {_RPN_FULL.fullmatch(current_decoded) is not None}",
                        #         f"\n  Regex prefix match? {_RPN_PREFIX.fullmatch(current_decoded) is not None}",
                        #         f"\n  Checker thinks regex_ok = {regex_ok}",
                        #           )
                        # # --- END DEBUGGING ---

                        if regex_ok and not allowed_by_bias:
                            mismatches.append((v, "should be ALLOWED but was BLOCKED by process_logits", decoded_candidate))
                        elif not regex_ok and allowed_by_bias:
                            mismatches.append((v, "should be BLOCKED but was ALLOWED by process_logits", decoded_candidate))

                    # Helpful debugging printouts only when there's a discrepancy
                    if mismatches:
                        print("\n[Regex bias check mismatches]")
                        for vid, msg, dec in mismatches[:20]:  # cap print volume
                            token_text = tokenizer.decode([vid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                            print(f"token id={vid} ('{token_text}'): {msg} | decoded answer='{dec}'")
                            for c in dec:
                                print(c, hex(ord(c)))
                        if len(mismatches) > 20:
                            print(f"  ... and {len(mismatches)-20} more mismatches")


            

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a constrained LLM with biased logits.")
    #parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset_name", type=str, default="nvidia/OpenCodeInstruct")
    parser.add_argument("--dataset_size", type=int, default=32,
                        help="Number of samples of the whole dataset to train on.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()



    test_finetune(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_size=args.dataset_size,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
