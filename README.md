![rabbitllm_logo](https://github.com/lyogavin/rabbitllm/blob/main/assets/rabbitllm_logo_sm.png?v=3&raw=true)

[**Quickstart**](#quickstart) | 
[**Configurations**](#configurations) | 
[**MacOS**](#macos) | 
[**Example notebooks**](#example-python-notebook) | 
[**FAQ**](#faq)

**RabbitLLM** optimizes inference memory usage, allowing 70B large language models to run inference on a single 4GB GPU card. No quantization, distillation, pruning or other model compression techniques that would result in degraded model performance are needed.

<a href="https://github.com/lyogavin/Anima/stargazers">![GitHub Repo stars](https://img.shields.io/github/stars/lyogavin/Anima?style=social)</a>
[![Downloads](https://static.pepy.tech/personalized-badge/rabbitllm?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/rabbitllm)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/wechat-Anima-brightgreen?logo=wechat)](https://static.aicompose.cn/static/wecom_barcode.png?t=1671918938)
[![Discord](https://img.shields.io/discord/1175437549783760896?logo=discord&color=7289da
)](https://discord.gg/2xffU5sn)
[![PyPI - RabbitLLM](https://img.shields.io/pypi/format/rabbitllm?logo=pypi&color=3571a3)
](https://pypi.org/project/rabbitllm/)
[![Website](https://img.shields.io/website?up_message=blog&url=https%3A%2F%2Fmedium.com%2F%40lyo.gavin&logo=medium&color=black)](https://medium.com/@lyo.gavin)
[![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dgavinli%26type%3Dpatrons&style=flat)](https://patreon.com/gavinli)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/lyogavin?logo=GitHub&color=lightgray)](https://github.com/sponsors/lyogavin)


## Updates

[2024/04/20] RabbitLLM supports Llama3 natively already. Run Llama3 70B on 4GB single GPU.

[2023/12/25] v2.8.2: Support MacOS running 70B large language models.

[2023/12/20] v2.7: Support RabbitLLMMixtral. 

[2023/12/20] v2.6: Added AutoModel, automatically detect model type, no need to provide model class to initialize model.

[2023/12/18] v2.5: added prefetching to overlap the model loading and compute. 10% speed improvement.

[2023/12/03] added support of **ChatGLM**, **QWen**, **Baichuan**, **Mistral**, **InternLM**!

[2023/12/02] added support for safetensors. Now support all top 10 models in open llm leaderboard.

[2023/12/01] rabbitllm 2.0. Support compressions: **3x run time speed up!**

[2023/11/20] rabbitllm Initial verion!

## Table of Contents

* [Quick start](#quickstart)
* [Model Compression](#model-compression---3x-inference-speed-up)
* [Configurations](#configurations)
* [Local model cache](#local-model-cache)
* [Run on MacOS](#macos)
* [Example notebooks](#example-python-notebook)
* [Supported Models](#supported-models)
* [Documentation](#documentation)
* [Acknowledgement](#acknowledgement)
* [FAQ](#faq)

## Quickstart

### 1. Install package

First, install the rabbitllm pip package.

```bash
pip install rabbitllm
```

**Optional — Flash Attention 2** (faster attention on Ampere+ GPUs, e.g. RTX 30xx/40xx):  
Install the `flash` extra so the default `attn_implementation="auto"` can use Flash when your system is compatible:

```bash
pip install rabbitllm[flash]
# or with uv (from source):
uv sync --extra flash
```

If that fails (no prebuilt wheel for your PyTorch/CUDA/Python), use a **prebuilt wheel** from [flashattn.dev](https://flashattn.dev) and install it: with **uv** run `uv pip install https://...whl` from the project root (no pip binary needed); otherwise `pip install https://...whl` in your venv. See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#flash-attention-installation-fails-build-from-source).

Without flash-attn installed, the model uses SDPA (PyTorch scaled dot-product attention). See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md#attention-implementation-flash-attention) for requirements and how to check if Flash is active.

### 2. Inference

Then, initialize RabbitLLMLlama2, pass in the huggingface repo ID of the model being used, or the local path, and inference can be performed similar to a regular transformer model.

(*You can also specify the path to save the splitted layered model through **layer_shards_saving_path** when init RabbitLLMLlama2.*

```python
from rabbitllm import AutoModel

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct")

# or use model's local path...
#model = AutoModel.from_pretrained("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
        'What is the capital of United States?',
        #'I like',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)

```
 
 
Note: During inference, the original model will first be decomposed and saved layer-wise. Please ensure there is sufficient disk space in the huggingface cache directory.
 

## Model Compression - 3x Inference Speed Up!

We just added model compression based on block-wise quantization-based model compression. Which can further **speed up the inference speed** for up to **3x** , with **almost ignorable accuracy loss!** (see more performance evaluation and why we use block-wise quantization in [this paper](https://arxiv.org/abs/2212.09720))

![speed_improvement](https://github.com/lyogavin/Anima/blob/main/assets/rabbitllm2_time_improvement.png?v=2&raw=true)

#### How to enable model compression speed up:

* Step 1. make sure you have [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) installed by `pip install -U bitsandbytes `
* Step 2. make sure rabbitllm verion later than 2.0.0: `pip install -U rabbitllm` 
* Step 3. when initialize the model, passing the argument compression ('4bit' or '8bit'):

```python
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct",
                     compression='4bit' # specify '8bit' for 8-bit block-wise quantization 
                    )
```

#### What are the differences between model compression and quantization?

Quantization normally needs to quantize both weights and activations to really speed things up. Which makes it harder to maintain accuracy and avoid the impact of outliers in all kinds of inputs.

While in our case the bottleneck is mainly at the disk loading, we only need to make the model loading size smaller. So, we get to only quantize the weights' part, which is easier to ensure the accuracy.

## Configurations
 
When initialize the model, we support the following configurations:

* **compression**: supported options: 4bit, 8bit for 4-bit or 8-bit block-wise quantization, or by default None for no compression
* **attn_implementation**: attention backend. Default **`"auto"`** (recommended): uses Flash Attention 2 when the `flash-attn` package is installed and the GPU is Ampere+ (compute capability ≥ 8.0), otherwise SDPA. You can force `"flash_attention_2"`, `"sdpa"`, or `"eager"`. Install with `pip install rabbitllm[flash]` to enable Flash when compatible.
* **profiling_mode**: supported options: True to output time consumptions or by default False
* **layer_shards_saving_path**: optionally another path to save the splitted model
* **token** (or **hf_token**): Hugging Face token for gated repos (e.g. *meta-llama/Llama-2-7b-hf*). Prefer `token` for new code (required in transformers v5).
* **prefetching**: prefetching to overlap the model loading and compute. By default, turned on. For now, only RabbitLLMLlama2 supports this.
* **delete_original**: if you don't have too much disk space, you can set delete_original to true to delete the original downloaded hugging face model, only keep the transformed one to save half of the disk space. 

## Local model cache

To avoid re-downloading models and keep the cache **out of the repository** (not committed to git), use one of these options.

**Option 1 — Cache inside the project (recommended)**  
The repo has a `models/` directory for this; it is in `.gitignore`, so nothing you put there is committed. Before running scripts or notebooks from the repo root:

```bash
export HF_HOME="$(pwd)/models"
```

Downloads and RabbitLLM split layers will then live under `./models/` and won’t be re-downloaded. To load from that path later, use the local path: `AutoModel.from_pretrained("./models/...")` (exact subdir name depends on Hugging Face, e.g. `models--org--repo-name`).

**Option 2 — Global cache**  
By default Hugging Face uses `~/.cache/huggingface/hub/`. The first download goes there and later runs reuse it; that path is outside the repo. To use a different global cache: `export HF_HOME=~/.cache/rabbitllm`.

**Summary:** Use `export HF_HOME="$(pwd)/models"` for a project-local cache; `models/` and `.models/` are already in `.gitignore`.

## MacOS

Just install rabbitllm and run the code the same as on linux. See more in [Quick Start](#quickstart).

* make sure you installed [mlx](https://github.com/ml-explore/mlx?tab=readme-ov-file#installation) and torch
* you probabaly need to install python native see more [here](https://stackoverflow.com/a/65432861/21230266)
* only [Apple silicon](https://support.apple.com/en-us/HT211814) is supported

Example [python notebook] (https://github.com/lyogavin/Anima/blob/main/air_llm/examples/run_on_macos.ipynb)


## Example Python Notebook

Example colabs here:

<a target="_blank" href="https://colab.research.google.com/github/lyogavin/Anima/blob/main/air_llm/examples/run_all_types_of_models.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### example of other models (ChatGLM, QWen, Baichuan, Mistral, etc):

<details>


* ChatGLM:

```python
from rabbitllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-base")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=True)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache= True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```

* QWen:

```python
from rabbitllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("Qwen/Qwen-7B")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```


* Baichuan, InternLM, Mistral, etc:

```python
from rabbitllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("baichuan-inc/Baichuan2-7B-Base")
#model = AutoModel.from_pretrained("internlm/internlm-20b")
#model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```


</details>


#### To request other model support: [here](https://docs.google.com/forms/d/e/1FAIpQLSe0Io9ANMT964Zi-OQOq1TJmnvP-G3_ZgQDhP7SatN0IEdbOg/viewform?usp=sf_link)

## Documentation

Technical notes for developers and contributors:

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Design: relationship with HuggingFace, tied weights and `lm_head`, KV cache (DynamicCache), attention implementations (eager, SDPA, FlashAttention 2).
- **[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)** — Transformers version, model compatibility matrix, **Flash Attention** (optional install, auto-detection), single-file checkpoints.
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — Common issues (zero logits, attention mask, cache errors) and how to debug.

## Acknowledgement

A lot of the code are based on SimJeg's great work in the Kaggle exam competition. Big shoutout to SimJeg:

[GitHub account @SimJeg](https://github.com/SimJeg), 
[the code on Kaggle](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag), 
[the associated discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446414).


## FAQ

### 1. MetadataIncompleteBuffer

safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer

If you run into this error, most possible cause is you run out of disk space. The process of splitting model is very disk-consuming. See [this](https://huggingface.co/TheBloke/guanaco-65B-GPTQ/discussions/12). You may need to extend your disk space, clear huggingface [.cache](https://huggingface.co/docs/datasets/cache) and rerun. 

### 2. ValueError: max() arg is an empty sequence

Most likely you are loading QWen or ChatGLM model with Llama2 class. Try the following:

For QWen model: 

```python
from rabbitllm import AutoModel #<----- instead of RabbitLLMLlama2
AutoModel.from_pretrained(...)
```

For ChatGLM model: 

```python
from rabbitllm import AutoModel #<----- instead of RabbitLLMLlama2
AutoModel.from_pretrained(...)
```

### 3. 401 Client Error....Repo model ... is gated.

Some models are gated models, needs Hugging Face API token. You can provide `token` (or `hf_token` for backward compatibility):

```python
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", token="HF_API_TOKEN")
```

### 4. ValueError: Asking to pad but the tokenizer does not have a padding token.

Some model's tokenizer doesn't have padding token, so you can set a padding token or simply turn the padding config off:

 ```python
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False  #<-----------   turn off padding 
)
```

## Citing RabbitLLM

If you find
RabbitLLM useful in your research and wish to cite it, please use the following
BibTex entry:

```
@software{rabbitllm2023,
  author = {Gavin Li},
  title = {RabbitLLM: scaling large language models on low-end commodity computers},
  url = {https://github.com/lyogavin/Anima/tree/main/air_llm},
  version = {0.0},
  year = {2023},
}
```


## Contribution 

Welcomed contributions, ideas and discussions!

If you find it useful, please ⭐ or buy me a coffee! 🙏

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://bmc.link/lyogavinQ)
