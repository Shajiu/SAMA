# SAMA: Structure-Aware Alignment and Modular Adaptation for Robust Low-Resource Multilingual Multimodal Understanding

**Official implementation of our paper published in IEEE Transactions on Multimedia (TMM).**

**Authors:** Jiu Sha, Yu Weng*, Kailiang Wen, Minhao Wei, Mengxiao Zhu*

---

### Structure-Aware Alignment and Modular Adaptation

SAMA is a unified framework designed to improve the robustness and generalization of Multimodal Large Language Models (MLLMs) in low-resource multilingual environments. It integrates three key components:

- **Structure-Aware Cross-Modal Alignment:** Explicitly models writing direction, script type, and reading order to improve visual-textual grounding.
- **Script-Conditioned Modular Adaptation:** Introduces script-grouped LoRA adapters with dynamic routing to facilitate cross-lingual transfer while mitigating catastrophic forgetting.
- **Multimodal Robustness Regularization:** Incorporates cross-lingual safety consistency and OCR-noise invariance objectives to improve robustness against noisy and adversarial multimodal inputs.

### Framework Overview

```mermaid
flowchart TB
    subgraph INPUT["Multimodal Inputs & Structural Encoding"]
        IMG["Input Image / Document"]
        TXT["Multilingual Text Instruction"]
        ATTR["Structural Attribute Estimation"]
        IMG --> ATTR
        TXT --> ATTR
        ATTR --> TOK["Structural Embeddings<br/>Direction / Script / Reading Order"]
    end

    subgraph ALIGN["Structure-Aware Cross-Modal Alignment"]
        VE["Visual Encoder & Projector<br/>(Frozen)"]
        TE["Text Representations"]
        TOK --> ALIGNMENT["Structure-Conditioned<br/>Visual-Text Alignment"]
        VE --> ALIGNMENT
        TE --> ALIGNMENT
        ALIGNMENT --> CONS["Cross-View Attention<br/>Consistency"]
    end

    subgraph ADAPT["Script-Conditioned Modular Adaptation"]
        ROUTER["Script-Level Router"]
        A["Group A: RTL Scripts"]
        B["Group B: Top-to-Bottom Scripts"]
        C["Group C: Mixed Layouts"]
        ROUTER --> A
        ROUTER --> B
        ROUTER --> C
        A --> LLM
        B --> LLM
        C --> LLM
        ALIGNMENT --> LLM["Frozen MLLM Backbone<br/>+ Active LoRA Adapter"]
    end

    subgraph OUTPUT["Multimodal Task Generation"]
        LLM --> CAP["Image Captioning"]
        LLM --> VQA["Visual Question Answering"]
        LLM --> TRANS["Vision-Guided Translation"]
        LLM --> SAFEOUT["Safe Response Generation"]
    end

    subgraph ROBUST["Multimodal Robustness Regularization"]
        NOISE["OCR-Noise Perturbation"]
        UNSAFE["Multilingual Unsafe Inputs"]
        NOISE --> RROB["OCR Robustness Loss"]
        UNSAFE --> RSAFE["Safety Consistency Loss"]
    end

    CONS --> TOTAL["Unified Multi-Task Training Objective"]
    CAP --> TASK["Task Supervision Loss"]
    VQA --> TASK
    TRANS --> TASK
    SAFEOUT --> TASK
    RROB --> TOTAL
    RSAFE --> TOTAL
    TASK --> TOTAL
    CONS --> TOTAL
```

### Training

SAMA adopts a unified multi-task optimization strategy, jointly optimizing task supervision, structural alignment, cross-lingual safety consistency, and OCR robustness.

**Training Pipeline**

1. Load the pretrained Qwen-2.5-VL backbone.
2. Initialize structural embeddings and script-conditioned LoRA modules.
3. Load multilingual multimodal training samples and structural metadata.
4. Apply structural consistency and OCR-noise perturbations.
5. Perform multi-task supervised training with auxiliary robustness objectives.
6. Save the model checkpoint and modular adapter parameters.

**Training Command**

The following is an illustrative command. The actual script and arguments will be updated with the official code release.

```bash
BASE_MODEL=/path/to/Qwen2.5-VL-7B-Instruct
TRAIN_DATA=/path/to/SAMA_train
OUTPUT_DIR=/path/to/SAMA-7B

python <TRAINING_SCRIPT> \
    --model_name_or_path ${BASE_MODEL} \
    --train_data ${TRAIN_DATA} \
    --output_dir ${OUTPUT_DIR}
```

### Inference

During inference, SAMA automatically estimates structural attributes and activates the corresponding script-conditioned adapter.

**Inference Pipeline**

1. **Input Processing:** Receive an image and multilingual instruction.
2. **Structural Estimation:** Identify script type, writing direction, and dominant reading order using Unicode ranges and language identification signals.
3. **Adapter Routing:** Activate the corresponding script-conditioned LoRA branch.
4. **Multimodal Reasoning:** Perform structure-conditioned visual-textual reasoning.
5. **Response Generation:** Generate the requested multilingual output.

For mixed-script samples, the dominant script and reading flow are determined according to the largest text-region proportion.

**Inference Command**

The following is an illustrative command. The actual implementation will be updated upon code release.

```bash
MODEL_PATH=/path/to/SAMA-7B
IMAGE_PATH=/path/to/example.jpg

python <INFERENCE_SCRIPT> \
    --model_path ${MODEL_PATH} \
    --image_path ${IMAGE_PATH} \
    --language ug \
    --task vqa \
    --prompt "What is shown in this image?"
```

### Repository Structure

```text
SAMA/
│
├── configs/
│   ├── sama_3b.yaml
│   └── sama_7b.yaml
│
├── data/
│   ├── README.md
│   └── examples/
│
├── models/
│   └── README.md
│
├── scripts/
│   ├── train.sh
│   └── inference.sh
│
├── sama/
│   ├── modeling/
│   ├── alignment/
│   ├── adapters/
│   ├── routing/
│   └── robustness/
│
├── train.py
├── inference.py
├── evaluate.py
├── requirements.txt
└── README.md
```

### Model Weights and Data Availability

| Resource | Availability |
|---|---|
| Source Code | To be released |
| SAMA-3B Checkpoint | To be updated |
| SAMA-7B Checkpoint | To be updated |
| Training Dataset | Access information to be updated |
| Evaluation Dataset | Access information to be updated |
| Evaluation Scripts | To be released |

Please refer to this repository for the latest release information.

### Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{sha2026sama,
  title={SAMA: Structure-Aware Alignment and Modular Adaptation for Robust Low-Resource Multilingual Multimodal Understanding},
  author={Sha, Jiu and Weng, Yu and Wen, Kailiang and Wei, Minhao and Zhu, Mengxiao},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}
```

---

**HeTong Multilingual Intelligence Team**

*Making Multilingual AI Understandable, Trustworthy, and Accessible.*
