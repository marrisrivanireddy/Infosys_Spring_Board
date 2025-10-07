# Milestone 1 
#  Milestone 1 – Code Understanding and Model Comparison

##  Problem Statement
The goal of **Milestone 1** is to analyze multiple Python code snippets using advanced NLP and deep learning techniques.  
We aim to understand how different transformer-based models interpret and represent the semantics of source code.  

###  Objectives
- **AST (Abstract Syntax Tree) parsing** to extract structure and logical patterns  
- **Tokenization** to convert code into machine-readable input  
- **Encoding with Pretrained Models** (MiniLM, DistilRoBERTa, MPNet)  
- **Model Comparison** based on semantic similarity and conceptual focus  
- **Visualization** of how models differ in understanding programming logic  

---

##  Models Used

| Model | Description |
|--------|--------------|
| **MiniLM** | A lightweight transformer producing high-quality embeddings for text and code. Optimized for semantic similarity tasks. |
| **DistilRoBERTa** | A distilled version of RoBERTa that retains 95% of its accuracy while being faster and smaller. Great for contextual comprehension. |
| **MPNet** | Combines Masked and Permuted language modeling to capture both position and meaning efficiently. Excellent for general-purpose understanding. |

>  Note: These models **don’t generate human-like text outputs**.  
> They only **understand semantics** and represent meaning as embeddings (numerical vectors) used for similarity comparison.

---
## Flow of the Process
CODE SNIPPETS
     ↓
AST PARSING → Extract structure, patterns, logic
     ↓
TOKENIZATION → Convert into model-readable tokens
     ↓
ENCODING → Models generate embeddings (MiniLM, DistilRoBERTa, MPNet)
     ↓
EXPLANATION → Models describe focus areas via semantic similarity
     ↓
COMPARISON → Analyze which model best understands each snippet
     ↓
VISUALIZATION → Heatmaps, bar charts, and word clouds

## Example Code Snippet

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

tep 1: Create Code Snippets

We generate 10 Python snippets covering a variety of programming constructs:

Simple functions

Classes and objects

Decorators

Async and await

Generators

Regex operations

Dataclasses

Functional programming

Context managers

List comprehensions

Each snippet is stored as a .py file inside the snippets/ folder.

🧮 Step 2: AST Parsing and Tokenization

Each code snippet is parsed using Python’s ast module to extract:

Function and class definitions

Imports and logical patterns

Code features like recursion, decorators, async, etc.

Then, we use pretrained tokenizers from each model to tokenize the snippets.
This allows models to “read” the code structure in token form.

Example Output:

{
  "functions": ["factorial"],
  "patterns": ["recursion"],
  "imports": [],
  "classes": []
}

🤖 Step 3: Model Encoding and Explanation

Each snippet is encoded using the following pretrained embedding models:

MiniLM

DistilRoBERTa

MPNet

Each model produces a dense vector embedding — a numerical representation of meaning.
We then compute cosine similarity between code embeddings and conceptual labels such as:

"code structure", "algorithmic logic", "data handling", "control flow"

This yields interpretive outputs like:

MiniLM focuses on: code structure (0.23), algorithmic logic (0.21), function documentation (0.19)
DistilRoBERTa focuses on: function documentation (0.25), control flow (0.20)
MPNet focuses on: data handling (0.27), object-oriented design (0.22)


These describe how each model perceives the snippet semantically.

📊 Step 4: Model Comparison and Visualization

Once we have the model outputs, we perform:

TF-IDF Vectorization – builds a shared vocabulary of all outputs

Cosine Similarity – measures overlap between model focus areas

Dominant Model Identification – finds which model best aligns per snippet

Visualization – generates:
🔥 Heatmaps showing inter-model similarity
📈 Bar charts for best model per snippet
☁️ Word clouds for key focus terms

🧾 Example Results
Snippet	MiniLM	DistilRoBERTa	MPNet	Best Model
factorial	0.93	0.89	0.91	MiniLM
fibonacci	0.91	0.88	0.89	MiniLM
dataclass	0.87	0.85	0.90	MPNet
stack class	0.86	0.91	0.89	DistilRoBERTa
🧭 Overall Analysis

After evaluating all 10 code snippets:

🧩 MiniLM performs best for algorithmic and logic-driven code.

🧮 MPNet excels at understanding structured, class-based snippets.

🗒️ DistilRoBERTa captures documentation and control flow semantics effectively.

Each model exhibits unique focus areas, showing that even pretrained models interpret code differently.

🌐 Extension: Real LLMs vs. Embedding Models
Aspect	Embedding Models (MiniLM, MPNet, DistilRoBERTa)	Real LLMs (GPT-4, GPT-5)
Purpose	Represent meaning numerically	Understand and generate text
Output	Vector embeddings	Natural language explanation
Functionality	Similarity, clustering, matching	Reasoning, explanation, summarization
Example Output	“Focus: algorithmic logic (0.91)”	“This function recursively computes factorial.”

Real LLMs combine semantic understanding + reasoning, allowing them to explain, debug, and refactor code — not just represent it.

🏁 Summary Table
Stage	Description	Output
AST Parsing	Extracts structural and logical elements	Function, class, import metadata
Tokenization	Converts code into token sequences	Tokens for each model
Encoding	Embeds semantic meaning	Dense vector representation
Explanation	Finds conceptual alignment	Model focus descriptions
Comparison	Evaluates similarity across models	Similarity matrices
Visualization	Displays understanding differences	Heatmaps, charts, and word clouds
📂 Repository Structure
Infosys_Spring_Board/
 └── Mile_Stone_1/
      ├── Main.py              # Full implementation of Milestone 1
      ├── README.md            # This documentation
      └── images/              # (Optional) Visual outputs like charts and clouds

💡 Conclusion

This milestone demonstrates how transformer-based models can analyze and understand code semantics.
By comparing embeddings and similarities, we reveal how different models focus on structure, logic, or documentation.

This project bridges static code analysis and natural language understanding, paving the way for AI-assisted programming tools that combine deep learning, reasoning, and interpretability.

✨ Author

👩‍💻 Marris Srivani Reddy
B.Tech – Computer Science (AI & ML Aligned Branch)
📘 Passionate about Software Development, Artificial Intelligence, and Data Science.


---

### ✅ Steps to Upload This README to GitHub
1. Copy everything above ⬆️ (starting from `# 🚀 Milestone 1 – Code Understanding and Model Comparison`)  
2. Go to your GitHub repository → `Mile_Stone_1/README.md`  
3. Paste the content into the editor  
4. Add a commit message:


Added final detailed Milestone 1 README

5. Click **Commit changes** ✅  

---

Would you like me to now **generate a flowchart image** (in a clear visual format like  
`Snippets → AST → Tokenization → Models → Comparison → Visualization`) so you can upload it inside `/im

