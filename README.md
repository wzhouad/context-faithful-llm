# Context-faithful Prompting for Large Language Models

Code and data for paper [Context-faithful Prompting for Large Language Models](https://arxiv.org/abs/2303.11315).

## How to Use

### Step 1:  Install Required Packages

Before you begin, make sure to install the necessary packages: ``openai``, ``scipy``, ``numpy``, ``tiktoken``, ``tqdm``, and ``scikit-learn``. To do so, run the following command:
``pip install -r requirements.txt``.

### Step 2: Download the Datasets

Download the NQ and RealtimeQA datasets from [Google Drive](https://drive.google.com/file/d/1DJ1ajmLNAKVTBWnM7SkP93EYQ2cav3Mk/view?usp=sharing) and extract them to the repository folder. Please note that the TACRED dataset is not included due to its LDC license.

### Step 3: Add Your OpenAI API Key

Insert your OpenAI API key to ``api_secrets.py``.

### Step 4: Run Experiments

Run experiments on the NQ dataset in the knowledge conflict setting using the following command:
`` python knowledge_conflict.py --schema ${SCHEMA} --demo_mode ${DEMO_MODE}``

To perform experiments on the RealTime QA dataset in the abstention setting, use this command:
``python abstention.py --schema ${SCHEMA} --demo_mode ${DEMO_MODE}``

The ``SCHEMA`` parameter refers to the prompting templates described in the paper and can take the following values: ``base``, ``attr``, ``instr``, ``opin``, or ``instr+opin``. The ``DEMO_MODE`` parameter represents the demonstration method, with possible values being ``none`` (zero-shot), ``counter`` (counterfactual demonstrations, applicable only in the knowledge conflict setting), and ``original`` (original demonstrations).

**Please be aware that running experiments can be costly. Few-shot evaluation on the full dataset is estimated to cost around $150 for NQ and $30 for RealTime QA when using the ``text-davinci-003`` engine for each prompting templates.**

## Citation
```bibtex
@article{zhou2023context,
  title={Context-faithful Prompting for Large Language Models},
  author={Zhou, Wenxuan and Zhang, Sheng and Poon, Hoifung and Chen, Muhao},
  journal={arXiv preprint arXiv:2303.11315},
  year={2023}
}
```
