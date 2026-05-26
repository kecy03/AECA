<h1 align="center">
Rethinking Pretraining Data Detection for LLMs:<br>
From Local to Global [ICML 2026]
</h1>

<p align="center">
  <a href="#">📖 Paper</a> |
  <a href="#">📸 Poster</a> |
  <a href="#">🏠 Author Homepage</a>
</p>

<p align="center">
State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China
</p>

This repository focuses on the problem of pretraining data detection for LLMs and proposes a novel method based on the **probability dynamics of tokens during the text generation process.**

## 📌 Overview
Modern LLMs owe their success to massive training datasets. However, the use of such extensive, unchecked data raises serious issues like **privacy leakage** and **data contamination**. Consequently, the ability to audit whether a target text belongs to the pretraining corpus is essential for trustworthy AI.

We propose <u>**A**</u>daptive <u>**E**</u>ntropic <u>**C**</u>onvolutional <u>**A**</u>nalysis **(AECA)**, a novel pretraining data detection framework.

<p align="center">
  <img src="fig/AECA.png" alt="AECA framework" width="100%">
</p>

## 🔥 News

- [2026/05] We release our code.
- [2026/05] Our paper is accepted to [ICML 2026](https://icml.cc/Conferences/2026)! 🎉
- [2026/01] Our paper is under review.

## 🛠️ Requirements and Dependencies

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## 🗂️ Datasets
The datasets used in our experiments can be accessed as follows:
* **WikiMIA:** Available at [🤗swj0419/WikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA).
* **MIMIR:** Available at [🤗iamgroot42/mimir](https://huggingface.co/datasets/iamgroot42/mimir).
* **Reference Corpus $\mathcal{D}_{\text{ref}}$:** Available at [🤗allenai/c4](https://huggingface.co/datasets/allenai/c4).

Download with Hugging Face CLI:

```bash
hf download swj0419/WikiMIA --repo-type dataset --local-dir data/WikiMIA
```

## 🤖 Models
You can download the models used in our experiments from Hugging Face. For example:

* **Pythia-2.8B:** [🤗EleutherAI/pythia-2.8b](https://huggingface.co/EleutherAI/pythia-2.8b).
* **GPT-NeoX-20B:** [🤗EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b)

Download with Hugging Face CLI:

```bash
hf download EleutherAI/pythia-2.8b --local-dir models/pythia-2.8b
```

## 🚀 Running
Follow the steps below to reproduce the results:

1. First, configure the paths for the datasets and models in `config.py`.

2. Calculate the token frequency of the reference corpus:
    ```bash
    python build_freq.py
    ```

3. Finally, run the AECA evaluation script:
    ```bash
    python AECA.py
    ```

## 🙏 Acknowledgement

We sincerely thank the following GitHub repositories for their valuable codebases used as baselines:

- [Min-K% Prob](https://github.com/swj0419/detect-pretrain-code): Uses the k% tokens with the minimum likelihood for score computation.
- [Min-K%++ Prob](https://github.com/zjysteven/mink-plus-plus): Uses the k% tokens with the minimum normalized likelihood for score computation.
- [DC-PDD](https://github.com/zhang-wei-chao/DC-PDD): Uses the frequency distribution of a large corpus to calibrate token probabilities.
- [PAC](https://github.com/yyy01/PAC%23%23-Detect-Data-Contamination-with-PAC): Measures the polarization distance of model confidence between the original input and adjacent samples generated via random swap augmentation.

We also thank [TimeDART](https://github.com/ustc-time-series/TimeDART) for the `README.md` writing template.


## ⭐ Citation

If you find our work helpful, please consider giving this repository a star and citing our work.

```bibtex
TBD
```


