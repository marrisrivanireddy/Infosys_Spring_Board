#  Infosys CodeGenie AI — Milestone 2  
###  Code Generation and Model Comparison using Multiple LLMs

---

##  **Problem Statement**

Develop an **AI-powered Code Generator and Explainer** capable of generating programming code from natural language prompts using multiple open-source language models.  

The system should:
- Generate high-quality, executable code for diverse programming tasks.
- Compare multiple LLMs based on accuracy, output quality, and latency.
- Provide interactive UIs for users to test prompts and visualize performance.

---

##  **Models Used**

| # | Model | Hugging Face ID | Parameters | Description |
|---|--------|----------------|-------------|--------------|
| 1 | **DeepSeek-Coder-1.3B** | deepseek-ai/deepseek-coder-1.3b-instruct | 1.3B | Lightweight code model optimized for instruction-based generation. |
| 2 | **Phi-2** | microsoft/phi-2 | 2.7B | A small, efficient model with strong logical and mathematical reasoning. |
| 3 | **Gemma-2B-IT** | google/gemma-2b-it | 2B | Instruction-tuned variant of Google’s Gemma, trained on multi-lingual datasets. |
| 4 | **Stable-Code-3B** | stabilityai/stable-code-3b | 3B | Focused on reliable code synthesis and better language diversity. |
| 5 | **Replit-Code-3B** | replit/replit-code-v1-3b | 3B | Designed for end-to-end software generation and completion tasks. |

---
##  Flow of the Process

GPU CHECK AND ENVIRONMENTSETUP

↓  
hugging face authentication

↓  
INDIVIDUAL MODEL CHECK  

↓  
EVALUATIONG THE LOGIC ACCURACY 

↓  
COMPAROSION OF MODELS

↓  
LAUNCHING THE UI

## INDIVIDUAL MODEL CHECK

### Model 5 — Replit-Code-3B
<p align="center" style="vertical-align: top;"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/r1.png" alt="Replit-Code Output" width="48%" height="300px" style="margin-right:15px; vertical-align: top; object-fit: contain;"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/r2.png" alt="Replit-Code Accuracy" width="48%" height="300px" style="vertical-align: top; object-fit: contain;"> </p>

### Model 2 — Phi-2 (2.7B)
<p align="center" style="vertical-align: top;"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/phi-1.png" alt="Phi-1 Output" width="48%" height="300px" style="margin-right:15px; vertical-align: top; object-fit: contain;"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/phi-2.png" alt="Phi-2 Output" width="48%" height="300px" style="vertical-align: top; object-fit: contain;"> </p>

### Model 3 — Gemma-2B-IT (2B)
<p align="center" style="vertical-align: top;"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/g1.png" alt="Gemma-2B Output" width="48%" height="300px" style="margin-right:15px; vertical-align: top; object-fit: contain;"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/g2.png" alt="Gemma-2B Accuracy" width="48%" height="300px" style="vertical-align: top; object-fit: contain;"> </p>

### Model 4 — Stable-Code-3B

<p align="center" style="vertical-align: top;"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/s1.png" alt="Stable-Code Output" width="48%" height="300px" style="margin-right:15px; vertical-align: top; object-fit: contain;"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/s2.png" alt="Stable-Code Accuracy" width="48%" height="300px" style="vertical-align: top; object-fit: contain;"> </p>



## COMPAROSION OF MODELS - Overall Performance


<p align="center" style="vertical-align: top;"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/c1.png" alt="Model Comparison Graph 1" width="48%" height="300px" style="margin-right:15px; vertical-align: top; object-fit: contain;"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/c2.png" alt="Model Comparison Graph 2" width="48%" height="300px" style="vertical-align: top; object-fit: contain;"> </p>

### Model Evaluation — Ranking Based on Performance
<p align="center"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/comp.png" alt="Model Ranking Graph" width="90%" height="400px" style="object-fit: contain; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"> </p>

### USER INTERFACE

### UI 1
<p align="center"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/ui1.png" alt="UI 1" width="95%" height="250px" style="object-fit: cover; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);"> </p>

### UI 2
<p align="center"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/ui2.png" alt="UI 2" width="95%" height="250px" style="object-fit: cover; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);"> </p>

### UI 3
<p align="center"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/ui3.png" alt="UI 3" width="95%" height="250px" style="object-fit: cover; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);"> </p>

### UI 4
<p align="center"> <img src="https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/Mile_Stone_2/ui4.png" alt="UI 4" width="95%" height="250px" style="object-fit: cover; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);"> </p>


#  CodeGenieAI — Model Comparison & Analysis (Milestone 2)

This project is part of **Infosys CodeGenieAI (Milestone 2)**, focusing on **code generation and explanation** using multiple open-source models from Hugging Face.  
The objective is to **evaluate**, **compare**, and **visualize** the performance of these models across various programming domains.

---

##  Models Evaluated

| Model Name | Developer | Parameters | Type | Key Strength |
|-------------|------------|-------------|------|---------------|
| **DeepSeek-Coder (1.3B)** | DeepSeek AI | 1.3B | Instruction-tuned code model | Fast inference, good syntax accuracy |
| **Phi-2 (2.7B)** | Microsoft | 2.7B | General-purpose LLM | Strong reasoning and natural language coherence |
| **Gemma-2B-IT** | Google | 2B | Instruction-tuned model | Optimized for text + code tasks |
| **Stable-Code (3B)** | Stability AI | 3B | Code generation model | Handles long code contexts efficiently |
| **Replit-Code (3B)** | Replit | 3B | Code generation (multi-language) | Good for real-world dev use cases and multi-file code |

---

##  Differences Between the Models

| Aspect | DeepSeek-Coder | Phi-2 | Gemma-2B-IT | Stable-Code | Replit-Code |
|:--|:--|:--|:--|:--|:--|
| **Training Data** | Primarily GitHub code | Mixed code + text reasoning | Fine-tuned with instruction datasets | Long code sequences (Python, JS, C++) | Real-world dev data from Replit users |
| **Output Style** | Compact, structured | Detailed, explanation-rich | Balanced between code & text | Long, optimized solutions | Practical, production-oriented |
| **Response Speed** | ⚡ Very fast | ⚙️ Moderate | 🚀 Fast | 🕐 Slightly slow | ⚙️ Moderate |
| **Accuracy on Syntax** | High | Medium | High | High | Medium |
| **Language Diversity** | Python, C++, Java | Text + limited code | Python, JS, Java | Python, JS, Rust | Python, JS, Web stack |
| **Explainability** | Limited comments | Very strong explanations | Balanced | Low | High clarity but less reasoning |
| **Best Use Case** | Competitive coding | Code understanding + NLP tasks | Learning/explaining code | Long function generation | Full project code generation |

---

##  Limitations of Each Model

###  DeepSeek-Coder (1.3B)
-  Limited reasoning for high-level algorithmic explanations  
-  May skip comments or docstrings  
-  Best for concise, clean code snippets  

---

###  Phi-2 (2.7B)
-  Slower inference time  
-  Sometimes generates incomplete code blocks  
-  Excellent at combining natural language reasoning with coding logic  

---

###  Gemma-2B-IT
-  Slightly generic outputs for domain-specific prompts  
-  Consistent formatting and code quality  
-  Performs well in explainable code generation tasks  

---

###  Stable-Code (3B)
-  Requires more memory (GPU/RAM)  
-  May generate redundant lines in large scripts  
-  Excels at large function generation and contextual understanding  

---

###  Replit-Code (3B)
-  Occasionally overfits to “web app” or “JS-like” syntax  
-  Produces slower completions  
-  Practical and realistic coding style suitable for deployment  

---

##  Conclusion

Each model offers unique trade-offs between **speed**, **accuracy**, **reasoning**, and **code complexity**.  
After testing and evaluation:

-  **DeepSeek-Coder (1.3B)** → Best for **fast, short, and syntactically accurate** code.  
-  **Phi-2 (2.7B)** → Best for **reasoning-driven and explanation-rich** tasks.  
-  **Gemma-2B-IT** → Most **balanced** for generating and explaining code together.  
-  **Stable-Code (3B)** → Ideal for **long, multi-function code generation**.  
-  **Replit-Code (3B)** → Great for **real-world project-style outputs**.

---

###  Final Verdict

>  *No single model is universally best — the ideal choice depends on your task:*  
> - Choose **DeepSeek-Coder** for quick results  
> - Choose **Phi-2** for strong logic and reasoning  
> - Choose **Gemma-2B-IT** for teaching or learning tasks  
> - Choose **Stable-Code** for long programs  
> - Choose **Replit-Code** for realistic, deployment-style projects

---

