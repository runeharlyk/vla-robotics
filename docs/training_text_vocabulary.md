# Training Text Token Vocabulary

Generated on 2026-05-05.

This note measures the language-side vocabulary that appears in the public
training metadata we can reproduce for SmolVLA and the LIBERO SFT checkpoint.
It is about **unique tokenizer ids used by instruction/task text**, not action
tokens, image tokens, or the model's total tokenizer capacity.

## Method

The script is [`scripts/count_training_tokens.py`](../scripts/count_training_tokens.py).
It resolves the tokenizer from checkpoint `config.json`, reads LeRobot
`meta/tasks.*` files from Hugging Face dataset repos, appends the newline used
by the local SmolVLA wrapper, tokenizes with `max_length=48`, and excludes
padding/special tokens by default.

The script writes reproducible JSON summaries under `results/token_vocab/`.

```sh
$env:UV_CACHE_DIR='.uv-cache'
uv run python scripts/count_training_tokens.py --preset tokenizer --checkpoint lerobot/smolvla_base
uv run python scripts/count_training_tokens.py --preset community --checkpoint lerobot/smolvla_base
uv run python scripts/count_training_tokens.py --preset libero --checkpoint HuggingFaceVLA/smolvla_libero
uv run python scripts/count_training_tokens.py --preset libero_aggregate --checkpoint HuggingFaceVLA/smolvla_libero
```

## Results

| Stage/source | Tokenizer | Unique instruction strings | Unique token ids | Tokenizer coverage | Word-like strings |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolVLM2 tokenizer capacity, image checkpoint | `HuggingFaceTB/SmolVLM2-500M-Instruct` | - | 49,280 total tokenizer entries | 100% capacity | - |
| SmolVLM2 tokenizer capacity, video checkpoint | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | - | 49,280 total tokenizer entries | 100% capacity | - |
| SmolVLA community v1 task metadata | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 108 | 219 | 0.444% | 187 |
| SmolVLA community v2 task metadata | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 242 | 447 | 0.907% | 368 |
| SmolVLA community v1+v2 task metadata | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 340 | 538 | 1.092% | 447 |
| LIBERO spatial | `HuggingFaceTB/SmolVLM2-500M-Instruct` | 10 | 29 | 0.059% | 26 |
| LIBERO object | `HuggingFaceTB/SmolVLM2-500M-Instruct` | 10 | 28 | 0.057% | 24 |
| LIBERO goal | `HuggingFaceTB/SmolVLM2-500M-Instruct` | 10 | 26 | 0.053% | 24 |
| LIBERO long | `HuggingFaceTB/SmolVLM2-500M-Instruct` | 10 | 49 | 0.099% | 45 |
| LIBERO all suites / `lerobot/libero` | `HuggingFaceTB/SmolVLM2-500M-Instruct` | 40 | 78 | 0.158% | 69 |

Artifacts:

- `results/token_vocab/smolvla_base_tokenizer.json`
- `results/token_vocab/smolvla_community_v1.json`
- `results/token_vocab/smolvla_community_v2.json`
- `results/token_vocab/smolvla_community.json`
- `results/token_vocab/libero_spatial.json`
- `results/token_vocab/libero_object.json`
- `results/token_vocab/libero_goal.json`
- `results/token_vocab/libero_long.json`
- `results/token_vocab/libero_suites.json`
- `results/token_vocab/libero_aggregate.json`

## Interpretation

The hypothesis is basically right for the VLA adaptation stages: the language
surface used by public SmolVLA robotics metadata is small, and the LIBERO SFT
surface is very small. LIBERO all-suites uses only 40 task strings and 78 unique
SmolVLM2 tokenizer ids. That is 0.158% of the tokenizer entries.

This does **not** mean the model only knows those 78 tokens. SmolVLA starts from
SmolVLM2, whose public model cards list broad image/video/text instruction
datasets such as The Cauldron, Docmatix, LLaVA-OneVision-Data, M4-Instruct-Data,
FineVideo, MAmmoTH-VL-Instruct, LLaVA-Video-178K, Video-STaR, Vript, VISTA-400K,
MovieChat, and ShareGPT4Video. The exact unique token ids used during SmolVLM2
training cannot be recovered from the model card alone; reproducing that number
requires the full original training mixture and preprocessing.

The practical concern is therefore not unknown tokenizer ids. It is that the
robotics fine-tuning stages bind a tiny set of task phrasings to continuous
actions. New instructions usually tokenize into known SmolVLM2 tokens, but they
can still be out of distribution for the action expert because the VLA stage saw
few robot-language combinations.

## Source Notes

- `lerobot/smolvla_base` config resolves to
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `tokenizer_max_length=48`.
- `HuggingFaceVLA/smolvla_libero` config resolves to
  `HuggingFaceTB/SmolVLM2-500M-Instruct`, `tokenizer_max_length=48`.
- The public `HuggingFaceVLA/community_dataset_v1` card says it was used to
  pretrain SmolVLA, after filtering and manual task-description curation.
- `HuggingFaceVLA/community_dataset_v2` expands the same community-data format.
- `HuggingFaceVLA/smolvla_libero` declares `lerobot/smolvla_base` as its base
  model but the card has `datasets: unknown`; this repo's configs and jobs use
  `lerobot/libero` or the four concrete LIBERO suite repos for SFT/evaluation.

Relevant public pages:

- https://huggingface.co/lerobot/smolvla_base
- https://huggingface.co/HuggingFaceVLA/smolvla_libero
- https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct
- https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v1
- https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v2
- https://huggingface.co/datasets/lerobot/libero
