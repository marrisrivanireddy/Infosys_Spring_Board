# 🚀 Infosys CodeGenie AI — Milestone 2  
### 💡 Code Generation and Model Comparison using Multiple LLMs

---

## 🧩 **Problem Statement**

Develop an **AI-powered Code Generator and Explainer** capable of generating programming code from natural language prompts using multiple open-source language models.  

The system should:
- Generate high-quality, executable code for diverse programming tasks.
- Compare multiple LLMs based on accuracy, output quality, and latency.
- Provide interactive UIs for users to test prompts and visualize performance.

---

## 🧠 **Models Used**

| # | Model | Hugging Face ID | Parameters | Description |
|---|--------|----------------|-------------|--------------|
| 1 | **DeepSeek-Coder-1.3B** | `deepseek-ai/deepseek-coder-1.3b-instruct` | 1.3B | Lightweight code model optimized for instruction-based generation. |
| 2 | **Phi-2** | `microsoft/phi-2` | 2.7B | A small, efficient model with strong logical and mathematical reasoning. |
| 3 | **Gemma-2B-IT** | `google/gemma-2b-it` | 2B | Instruction-tuned variant of Google’s Gemma, trained on multi-lingual datasets. |
| 4 | **Stable-Code-3B** | `stabilityai/stable-code-3b` | 3B | Focused on reliable code synthesis and better language diversity. |
| 5 | **Replit-Code-3B** | `replit/replit-code-v1-3b` | 3B | Designed for end-to-end software generation and completion tasks. |

---

## 🔄 **Workflow / Flowchart**

Below is the step-by-step workflow implemented in Colab:

```mermaid
graph TD;
    A[Start] --> B[Environment Setup & GPU Check]
    B --> C[Hugging Face Authentication]
    C --> D[Define 10 Programming Prompts]
    D --> E[Run Each Model Sequentially]
    E --> F[Generate Code Outputs]
    F --> G[Evaluate Logic Accuracy & Latency]
    G --> H[Compare Results (CSV + Graphs)]
    H --> I[Rank Models Based on Final Score]
    I --> J[Launch UI-1: Single Model Code Generator]
    J --> K[Launch UI-2: Multi-Model Comparison Studio]
    K --> L[End]

