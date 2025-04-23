<div align="center">

# _DeepMath_

<div>
🗄️ A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning
</div>
</div>

<div>
<br>

<div align="center">

[![Data](https://img.shields.io/badge/Data-4d5eff?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
[![Github](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/zwhe99/DeepMath)
[![GitHub Stars](https://img.shields.io/github/stars/zwhe99/DeepMath?style=for-the-badge&logo=github&logoColor=white&label=Stars&color=000000)](https://github.com/zwhe99/DeepMath)
[![arXiv](https://img.shields.io/badge/arXiv-2504.11456-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2504.11456)
</div>
</div>

## 🔥 News

- **April 14, 2025**: We release **`DeepMath-103K`**, a large-scale dataset featuring challenging, verifiable, and decontaminated math problems tailored for RL and SFT. We open source:
  - 🤗 Training data: [`DeepMath-103K`](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
  - 💻 Code: [`DeepMath`](https://github.com/zwhe99/DeepMath)
  - 📝 Paper detailing data curation: [`arXiv:2504.11456`](https://arxiv.org/abs/2504.11456)
  - *(Coming Soon)*: Model weights trained on DeepMath-103K.



## 📖 Overview

**`DeepMath-103K`** is meticulously curated to push the boundaries of mathematical reasoning in language models. Key features include:

**1. Challenging Problems**: DeepMath-103K has a strong focus on difficult mathematical problems (primarily Levels 5-9), significantly raising the complexity bar compared to many existing open datasets.

<div align="center"> <img src="./assets/github-difficulty.png" width="90%"/>

<sub>Difficulty distribution comparison.</sub> </div>

**2. Broad Topical Diversity**: The dataset spans a wide spectrum of mathematical subjects, including Algebra, Calculus, Number Theory, Geometry, Probability, and Discrete Mathematics.

<div align="center"> <img src="./assets/github-domain.png" width="50%"/>

<sub>Hierarchical breakdown of mathematical topics covered in DeepMath-103K.</sub></div>

**3. Rigorous Decontamination**: Built from diverse sources, the dataset underwent meticulous decontamination against common benchmarks using semantic matching. This minimizes test set leakage and promotes fair model evaluation.

<div align="center"> <img src="./assets/github-contamination-case.png" width="80%"/>

<sub>Detected contamination examples. Subtle conceptual overlaps can also be identified.</sub> </div>

**4. Rich Data Format**: Each sample in `DeepMath-103K` is structured with rich information to support various research applications:

<div align="center"> <img src="./assets/github-data-sample.png" width="90%"/>

<sub>An example data sample from DeepMath-103K.</sub> </div>

- **Question**: The mathematical problem statement.
- **Final Answer**: A reliably verifiable final answer, enabling robust rule-based reward functions for RL.
- **Difficulty**: A numerical score for difficulty-aware training or analysis.
- **Topic**: Hierarchical classification for topic-specific applications.
- **R1 Solutions**: Three distinct reasoning paths from DeepSeek-R1, valuable for supervised fine-tuning (SFT) or knowledge distillation.

## 📊Main Results

We are currently training the `DeepMath-Zero-7B` and `DeepMath-1.5B` models using the `DeepMath-103K` dataset. These models are initialized from `Qwen2.5-7B-Base` and `R1-Distill-Qwen-1.5B`, respectively. The training process is ongoing.


|          Model           | MATH 500 |  AMC23   | Olympiad Bench | Miverva Math |  AIME24  |  AIME25  |
| :----------------------: | :------: | :------: | :------------: | :----------: | :------: | :------: |
|     Qwen2.5-7B-Base      |   54.8   |   35.3   |      27.8      |     16.2     |   7.7    |   5.4    |
|  Open-Reasoner-Zero-7B   |   81.8   |   58.9   |      47.9      |     38.4     |   15.6   |   14.4   |
| Qwen-2.5-7B-SimpleRL-Zoo |   77.0   |   55.8   |      41.0      |     41.2     |   15.6   |   8.7    |
|     DeepMath-Zero-7B     | **85.5** | **64.7** |    **51.0**    |   **45.3**   | **20.4** | **17.5** |

|          Model          | MATH 500 |  AMC23   | Olympiad Bench | Miverva Math |  AIME24  |  AIME25  |
| :---------------------: | :------: | :------: | :------------: | :----------: | :------: | :------: |
|  R1-Distill-Qwen-1.5B   |   84.7   |   72.0   |      53.1      |     36.6     |   29.4   |   24.8   |
| DeepScaleR-1.5B-Preview | **89.4** |   80.3   |    **60.9**    |   **42.2**   | **42.3** |   29.6   |
|  Still-3-1.5B-Preview   |   86.6   |   75.8   |      55.7      |     38.7     |   30.8   |   24.6   |
|   DeepMath-1.5B    |   89.0   | **81.6** |      60.1      |     40.6     |   39.8   | **30.8** |


## 🎯Quick Start

#### Environment Preparation

```shell
git clone --recurse-submodules https://github.com/zwhe99/DeepMath.git && cd DeepMath

conda create -y -n deepmath python=3.12.2 && conda activate deepmath
pip3 install ray[default]
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install omegaconf==2.4.0.dev3 hydra-core==1.4.0.dev1 antlr4-python3-runtime==4.11.0 vllm==0.7.3
pip3 install math-verify[antlr4_11_0]==0.7.0 fire deepspeed tensorboardX prettytable datasets transformers==4.49.0
pip3 install -e verl
```



#### Evaluation

```shell
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=XFORMERS VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 uni_eval.py \
    --base_model zwhe99/DeepMath-Zero-7B \
    --chat_template_name orz \
    --system_prompt_name simplerl \
    --output_dir  \
    --bf16 True \
    --tensor_parallel_size 8 \
    --data_id zwhe99/MATH \
    --split math500 \
    --max_model_len 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 16
```



#### Training

* Data Preparation

  ```shell
  DATA_DIR=/path/to/your/data
  python3 verl/examples/data_preprocess/deepmath_103k.py --local_dir $DATA_DIR
  ```

* Start Ray

  ```shell
  # Head node (×1)
  ray start  --head --port=6379  --node-ip-address=$HEAD_ADDR --num-gpus=8
  
  # Worker nodes (×7)
  ray start  --address=$HEAD_ADDR:6379 --node-ip-address=$WORKER_ADDR --num-gpus=8
  ```

* Launch training at head node. See `scripts/train` for training scripts.



## 🙏 Acknowledgements

This work can not be done without the help of the following works:

- **[verl](https://github.com/volcengine/verl)**: A very fast reinforcement learning framework.
- **[Vivacem/MMIQC](https://huggingface.co/datasets/Vivacem/MMIQC)**: A mixture of question-response pairs extracted from Mathematics Stack Exchange pages.
- **[TIGER-Lab/WebInstructSub](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub)**: Instruction data from MathStackExchange and ScienceStackExchange.
- **[AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)**: Approximately 860k math problems.



## 📚 Citation
```bibtex
@article{deepmath,
  title={DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and  Verifiable Mathematical Dataset for Advancing Reasoning},
  author={He, Zhiwei and Liang, Tian and Xu, Jiahao and Liu, Qiuzhi and Chen, Xingyu and Wang, Yue and Song, Linfeng and Yu, Dian and Liang, Zhenwen and Wang, Wenxuan and Zhang, Zhuosheng and Wang, Rui and Tu, Zhaopeng and Mi, Haitao and Yu, Dong},
  year={2025},
  eprint={2504.11456},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2504.11456}, 
}
```
