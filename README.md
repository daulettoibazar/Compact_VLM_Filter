<p align="center">
  <img src="images/header.png" alt="Trust the Model" width="800"/>
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

**Compact VLMs as In-Context Judge** introduces a lightweight, compact vision-language model (VLM) that acts as an **in-context judge** to filter noisy image-text data at scale — without relying on large, resource-intensive VLMs or costly external API calls.

> 🧠 Instead of stacking scoring modules on top of VLMs, we **test a small, purpose-built VLM** trained on high-quality data to internally assess alignment and quality of image/text pairs.

---

### 💡 Key Highlights

- 📦 **Compact VLM** used as an internal filtration function
- 💬 Improves **image/text alignment** and **caption fluency** (verified by image/text cosine similarity and caption perplexity scores)
- 🚫 No extra parameters, minimal overhead
- ⚖️ Competes with or outperforms larger, noisy datasets in downstream captioning performance

---

### 🛠️ How It Works

1. **Train a compact VLM** on curated image-caption datasets.
2. Use it as an **in-context evaluator** of image-text pairs.
3. Filter out misaligned, noisy, or linguistically poor samples (Set the appropriate threshold value during inference).
4. Use the filtered high-quality dataset for downstream VLM training.

---

### 🧪 Results

- Model trained of filtered, smaller dataset (only 18% of original data) rivaled or exceeded the quality of models trained on larger, unfiltered dataset.
- Demonstrated improvements in both **caption quality** and **data alignment**.

---

### 📂 Resources

- 📄 **[Paper (coming soon)]()**  
- 🤗 **[Model on HuggingFace](https://huggingface.co/Dauka-transformers/Compact_VLM_filter)**  
- 📁 **[Dataset Access](https://huggingface.co/datasets/Dauka-transformers/Tiny_VLM_filter_data)**  
- 🔧 **[Filtration model training scripts (Coming soon)]()**

---

### 🔧 Quickstart (Coming Soon)

```bash
# Clone the repo
git clone https://github.com/daulettoibazar/Compact_VLM_Filter.git
cd Compact_VLM_Filter

# Run evaluation
python scripts/filter_data.py --model_path HF_repo --output_image_folder "images_folder" --output_path "path to filtered json dataset" --dataset_path "Path to your original image/caption dataset json"

```
* Please make sure that your original image/caption dataset json file contains following keys: "image_path", "caption".


### 🤝 Acknowledgement

Special thanks to the **Qwen team** for open-sourcing the [Qwen2-VL](https://huggingface.co/Qwen/Qwen-VL) model series, which forms the backbone of this project.

We also gratefully acknowledge the creators of the following datasets and research works, which provided the foundation for generating and evaluating the samples in this dataset:

- **[pixparse/cc12m-wds](https://huggingface.co/datasets/pixparse/cc12m-wds)**  
  Based on the paper *Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts*  
  [Soravit Changpinyo, Piyush Sharma, Nan Ding, Radu Soricut, 2021](https://arxiv.org/abs/2102.08981).  

- **[UCSC-VLAA/Recap-COCO-30K](https://huggingface.co/datasets/UCSC-VLAA/Recap-COCO-30K)**  
  Derived from the paper *What If We Recaption Billions of Web Images with LLaMA-3?*  
  [Wang et al., 2024](https://arxiv.org/abs/2405.13587).

These datasets were instrumental in training our filtration-oriented small VLM.


### Citations

If you find this work useful, please consider citing:

```code
```