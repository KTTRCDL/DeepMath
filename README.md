<div align="center">

# DeepMath

<div>
🗄️ A Large-Scale, Challenging, Verifiable, and Decontaminated Mathematical Dataset for Advancing Reasoning
</div>
</div>

<div>
<br>

<div align="center">

[![Data](https://img.shields.io/badge/Data-4d5eff?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data)
[![Github](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/zwhe99/DeepMath)
[![GitHub Stars](https://img.shields.io/github/stars/zwhe99/DeepMath?style=for-the-badge&logo=github&logoColor=white&label=Stars&color=000000)](https://github.com/zwhe99/DeepMath)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/xxxx.xxxxx)
</div>
</div>

## 🔥 News

- **April 14, 2025**: We release **`DeepMath-103K`**, a large-scale dataset featuring challenging, verifiable, and decontaminated math problems tailored for SFT and RL. We open source:
  - 🤗 Training data: [`DeepMath-103K`](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
  - 💻 Code: [`DeepMath`](https://github.com/zwhe99/DeepMath)
  - 📝 Paper detailing data curation: [`arXiv:xxxx.xxxxx`](https://www.google.com/search?q=[https://arxiv.org/abs/xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx))
  - *(Coming Soon)*: Model weights trained on DeepMath.



## 📖 Overview

`DeepMath-103K` is meticulously curated to push the boundaries of mathematical reasoning in language models. Key features include:

**1. Challenging Problems**: DeepMath-103K has a strong focus on difficult mathematical problems (primarily Levels 5-9), significantly raising the complexity bar compared to many existing open datasets.

<div align="center"> <img src="./assets/github-difficulty.png" width="90%"/>

<sub>Difficulty distribution comparison (details in paper).</sub> </div>

**2. Broad Topical Diversity**: The dataset spans a wide spectrum of mathematical subjects, including Algebra, Calculus, Number Theory, Geometry, Probability, and Discrete Mathematics.

<div align="center"> <img src="./assets/github-domain.png" width="50%"/>

<sub>Hierarchical breakdown of mathematical topics covered in DeepMath-103K.</sub></div>

**4. Rigorous Decontamination**: Built from diverse sources, the dataset underwent meticulous decontamination against common benchmarks using semantic matching. This minimizes test set leakage and promotes fair model evaluation.

<div align="center"> <img src="./assets/github-contamination-case.png" width="80%"/>

<sub>An example data sample from DeepMath-103K.</sub> </div>

**5. Rich Data Format**: Each sample in `DeepMath-103K` is structured with rich information to support various research applications:

<div align="center"> <img src="./assets/github-data-sample.png" width="90%"/>
<sub>An example data sample from DeepMath-103K.</sub> </div>

- **Question**: The mathematical problem statement.
- **Final Answer**: A reliably verifiable final answer, enabling robust rule-based reward functions for RL.
- **Difficulty**: A numerical score for difficulty-aware training or analysis.
- **Topic**: Hierarchical classification for topic-specific applications.
- **R1 Solutions**: Three distinct reasoning paths from DeepSeek-R1, valuable for supervised fine-tuning (SFT) or knowledge distillation.
