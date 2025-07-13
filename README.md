<p align="center">
  <img src="images/logo.png" alt="Trust the Model" width="800"/>
</p>

<h1 align="center"> TRUST THE MODEL: Compact VLMs as In-Context Judges for Image-Text Data Quality</h1>

<p align="center">
  <a href="TODO: arxiv link"><img src="https://img.shields.io/badge/arXiv-Paper-blue" alt="arXiv Paper"></a>
  <a href="https://huggingface.co/Dauka-transformers/Compact_VLM_filter"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow" alt="HuggingFace Model"></a>
  <a href="https://huggingface.co/datasets/Dauka-transformers/Tiny_VLM_filter_data"><img src="https://img.shields.io/badge/HuggingFace-Dataset-yellow" alt="HuggingFace Dataset"></a>
  <a href="TODO: license link"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
</p>

---

### 🚀 Overview

**Compact VLMs as In-Context Judges** introduces a lightweight, compact vision-language model (VLM) that acts as an **in-context judge** to filter noisy image-text data at scale — without relying on large full-scale VLMs or extra filtering modules.

> 🧠 Instead of stacking filtration on top of large models, we **trust a small, purpose-built VLM** trained on high-quality data to internally assess alignment and linguistic fluency.

---

### 💡 Key Highlights

- 📦 **Compact VLM** used as an internal oracle
- 🔍 Filters noisy web data without auxiliary modules
- 💬 Improves **image/text alignment** and **caption fluency**
- 🚫 No extra parameters, minimal overhead
- ⚖️ Competes with or outperforms large web-crawled datasets

---

### 🛠️ How It Works

1. **Train a compact VLM** on curated image-caption datasets.
2. Use it as an **in-context evaluator** of image-text pairs.
3. Filter out misaligned, noisy, or linguistically poor samples.
4. Use the filtered high-quality dataset for downstream VLM training.

---

### 🧪 Results

- Our compact VLM rivaled or exceeded the quality of models trained on larger web-crawled data.
- Demonstrated improvements in both **caption quality** and **data alignment**.

---

### 📂 Resources

- 📄 **[Paper on arXiv](TODO)**  
- 🤗 **[Model on HuggingFace](TODO)**  
- 📁 **[Dataset Access (Coming Soon)](TODO)**  
- 🔧 **[Training Scripts](TODO)**

---

### 🔧 Quickstart (Coming Soon)

```bash
# Clone the repo
git clone https://github.com/yourname/vlm-judge
cd vlm-judge

# Download model
TODO

# Run evaluation
python judge.py --input data/images_and_captions.jsonl

```

### Citations

If you find this work useful, please consider citing:

```code
@article{your2025trust,
  title={TRUST THE MODEL: Compact VLMs as In-Context Judges for Image-Text Data Quality},
  author={YourName, et al.},
  journal={arXiv preprint arXiv:TODO},
  year={2025}
}
```
