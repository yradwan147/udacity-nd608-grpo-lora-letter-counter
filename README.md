# GRPO + LoRA: Teaching Qwen2.5-3B-Instruct to Count Letters

Submission for the **Udacity nd608 Generative AI Engineer** fundamentals
capstone (course `cd13303`).

We take `Qwen/Qwen2.5-3B-Instruct` and fine-tune it with **GRPO** (Group
Relative Policy Optimization), using **LoRA** adapters and **Unsloth + vLLM**
on a single GPU, to teach it the deceptively-hard task of:

> *"How many of the letter X are there in the word Y?"*

The model is taught to reason in a strict step-by-step format
(`<reasoning>` numbered spelling with running counts `</reasoning>`
`<answer>N</answer>`), and we verify that this new procedural reasoning skill
is added **without** forgetting general knowledge.

---

## Architecture

```
Qwen/Qwen2.5-3B-Instruct  (4-bit, frozen)
      │
      └─► LoRA adapters (r=32, q/k/v/o/gate/up/down_proj)
            │
            ├── GRPO Trainer (TRL + Unsloth + vLLM)
            │     │
            │     ├── REWARD_FUNCS (Cell 27):
            │     │     numbering_reward_func   (in-order numbered list)
            │     │     spelling_reward_func    (correct letter sequence)
            │     │     counting_reward_func    (accurate running count)
            │     │     format_reward_func      (<reasoning><answer> XML)
            │     │     correct_answer_reward_func (exact-match final integer)
            │     │
            │     └── Dataset built from ALL_WORDS (60 words × letters in & out of word)
            │
            └── grpo_saved_lora/  ← deliverable adapter
```

---

## Rubric coverage

### 1. Model Setup
* LoRA via `FastLanguageModel.get_peft_model` (Cell 4).
* `lora_rank = 32`, in the allowed `{8,16,32,64,128}` set.
* `target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` — all attention and MLP projections, as suggested by the rubric.
* Explanation comments in Cell 4 motivate both choices.

### 2. Baseline Prompting
* Cell 6 runs the model with an empty system prompt to demonstrate the failure mode.
* Cell 8 introduces a CoT system prompt with **one** explicit worked example (`room` / `o` → 2). The cell prints the model's step-by-step output, which is visibly improved over the baseline but still imperfect — motivating GRPO.

### 3. Reward Design & Validation
* Five reward functions cover the rubric's required dimensions:
  * `numbering_reward_func` (Cell 17): +0.5 for in-order numbering, -0.5 out-of-order, -1.0 for going past `len(word)`.
  * `spelling_reward_func` (Cell 19): +2.0 exact spelling, -0.5 per length diff, -1.0 per extra letter, -0.5 per missing letter.
  * `counting_reward_func` (Cell 21): +1.0 per accurate running count step, -1.0 otherwise, normalized.
  * `format_reward_func` (Cell 23): +0.5 for the XML shape, +0.5 if the extracted answer is a digit.
  * `correct_answer_reward_func` (Cell 25): +2.0 for exact-match final integer, -1.0 otherwise.
* Each cell ends with an `assert res[1] > res[0]` check that gives the better example a higher reward than the worse example — those asserts pass.

### 4. Training & Monitoring
* Cell 31: 5-step quick-pass training to validate the reward signal (non-zero rewards confirmed).
* Cell 34: 100-step "slow" training (longer than the quick pass, as required by the rubric).
* Cells 32 + 35: pandas + matplotlib plots of `reward` + `rewards/correct_answer_reward_func/mean` over training steps — both show an upward trend.

### 5. Final Comparison
* Cell 37 saves the LoRA adapter to `grpo_saved_lora/`.
* Cell 38 defines `compare_old_and_new_model(messages)` which runs vLLM inference both with and without the adapter.
* Cell 40 calls it on `ds[0]["prompt"]` — the NEW model produces the proper numbered spelling and final answer, the OLD model does not.
* Cell 43 checks catastrophic forgetting with *"What is the capital of the Philippines?"* — both models answer correctly, demonstrating that the new reasoning skill was learned without overwriting Qwen's base knowledge.

---

## How to reproduce

```bash
# 1) Build the venv (3.12) and install pinned deps. The Udacity-shipped
#    requirements.txt has triton==3.2.0 pinned but torch==2.7.1 requires
#    triton==3.3.1 — we bump that pin (see requirements.txt in this folder).
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt jupyter nbconvert

# 2) Execute the notebook end-to-end on a single GPU (A100 recommended,
#    V100 hit a Triton/vLLM LoRA-kernel compile bug on SM 7.0).
sbatch run_train.slurm
# or, interactively:
jupyter nbconvert --to notebook --execute gen_ai_fundamentals_project_starter.ipynb \
    --output executed.ipynb --ExecutePreprocessor.timeout=10800
```

---

## Files in this deliverable

```
deliverables/project1/
├── README.md                                       # this file
├── gen_ai_fundamentals_project_starter.ipynb       # completed notebook (TODOs filled)
├── executed_gen_ai_fundamentals_project_starter.ipynb  # same notebook with all outputs visible
├── grpo_saved_lora/                                # LoRA adapter saved after training
├── requirements.txt                                # the pinned-by-Udacity deps (with triton bump)
├── requirements_fixed.txt                          # the actual file used (triton==3.3.1)
├── run_train.slurm                                 # SLURM batch script
└── slurm_logs/                                     # stdout/stderr from the SLURM run
```
