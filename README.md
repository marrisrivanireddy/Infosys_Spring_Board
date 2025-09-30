# Milestone 1 — Code Explainer (Infosys Springboard Internship)

Welcome!   
This is my submission for **Milestone 1** of the Infosys Springboard internship.  
The task was to build a **Code Explainer pipeline** that can analyze Python code, understand its structure, and compare how different NLP models interpret it.  

---

##  What This Project Does
- Takes Python code snippets (12 covering functions, classes, recursion, async, etc.).  
- Breaks them down using **AST (Abstract Syntax Tree)**.  
- Splits them into tokens using Python’s `tokenize` module.  
- Sends them into three pretrained NLP models:  
  - MiniLM  
  - DistilRoBERTa  
  - MPNet  
- Compares the outputs and visualizes them with heatmaps and PCA plots.  

---

##  Model Outputs — How Each Model Sees the Code

###  MiniLM Similarity Heatmap
![MiniLM Heatmap](https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/images/minilm_heatmap.png?raw=true)  
 MiniLM groups recursive and iterative functions closely. Pretty efficient!  

###  DistilRoBERTa Similarity Heatmap
![DistilRoBERTa Heatmap](https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/images/roberta_heatmap.png?raw=true)  
 DistilRoBERTa struggles more since it’s trained mostly on natural text, not code.  

###  MPNet Similarity Heatmap
![MPNet Heatmap](https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/images/mpnet_heatmap.png?raw=true)  
 MPNet does the best job — it clusters similar algorithms together nicely.  

---

###  PCA Plots (2D View of Embeddings)

- **MiniLM PCA**  
  ![MiniLM PCA](https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/images/minilm_pca.png?raw=true)  
   You can see math-related snippets sitting close to each other.  

- **DistilRoBERTa PCA**  
  ![DistilRoBERTa PCA](https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/images/roberta_pca.png?raw=true)  
   The points are scattered, meaning it doesn’t see code similarity very well.  

- **MPNet PCA**  
  ![MPNet PCA](https://github.com/marrisrivanireddy/Infosys_Spring_Board/blob/main/images/mpnet_pca.png?raw=true)  
   The cleanest clusters — MPNet clearly “understands” code patterns better.  

---

## 📈 Do the Models Agree?
| Comparison                 | Agreement |
|-----------------------------|-----------|
| MiniLM vs MPNet            | 0.16 (weak) |
| MiniLM vs DistilRoBERTa    | 0.37 (moderate) |
| DistilRoBERTa vs MPNet     | 0.09 (very weak) |

 

---

