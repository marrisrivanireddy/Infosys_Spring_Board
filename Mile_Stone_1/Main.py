# ==========================================================
#  MILESTONE 1 — CODE EXPLAINER AND MODEL COMPARISON
# ==========================================================

# ---------- STEP 1 ----------
# Environment setup + create 10 Python code snippets (save to snippets/*.py and snippets.json)

# (1) Install required packages we'll need later (quiet mode)
!pip install -q sentence-transformers transformers torch nltk scikit-learn matplotlib pandas seaborn wordcloud

import os, json, textwrap

# (2) Create folder and sample snippets
os.makedirs('snippets', exist_ok=True)

snippets = [
# 1. Simple function
"""# snippet_01: simple function
def add(a, b):
    \"\"\"Return the sum of two numbers.\"\"\"
    return a + b
""",

# 2. Class with methods
"""# snippet_02: Stack class
class Stack:
    \"\"\"A simple stack implementation.\"\"\"
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        return self._items.pop()

    def __len__(self):
        return len(self._items)
""",

# 3. Generator (fibonacci)
"""# snippet_03: fibonacci generator
def fibonacci(n):
    \"\"\"Yield first n fibonacci numbers.\"\"\"
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
""",

# 4. Decorator example
"""# snippet_04: timing decorator
import time
def timing(fn):
    \"\"\"Decorator that measures execution time.\"\"\"
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print(f\"{fn.__name__} took {end-start:.6f}s\")
        return result
    return wrapper

@timing
def compute(n):
    s = 0
    for i in range(n):
        s += i
    return s
""",

# 5. Async example (simulated async)
"""# snippet_05: async example
import asyncio

async def async_wait_and_return(x, delay=0.1):
    \"\"\"Simulated async task.\"\"\"
    await asyncio.sleep(delay)
    return f\"done:{x}\"

# Example usage (not executed here):
# asyncio.run(async_wait_and_return(5))
""",

# 6. dataclass usage
"""# snippet_06: dataclass example
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

    def greet(self):
        return f\"Hi, I'm {self.name} and I'm {self.age}.\"
""",

# 7. Recursive + memoization
"""# snippet_07: factorial with memo
from functools import lru_cache

@lru_cache(maxsize=None)
def factorial(n):
    \"\"\"Return n! recursively with memoization.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n-1)
""",

# 8. Regex example
"""# snippet_08: regex find emails
import re

def find_emails(text):
    \"\"\"Return list of email-like tokens in text.\"\"\"
    pattern = r\"[\\w\\.-]+@[\\w\\.-]+\\.[A-Za-z]{2,}\"
    return re.findall(pattern, text)
""",

# 9. Context manager example
"""# snippet_09: context manager for files
from contextlib import contextmanager

@contextmanager
def open_file(path, mode='r'):
    f = open(path, mode)
    try:
        yield f
    finally:
        f.close()
""",

# 10. Functional + list comprehension example
"""# snippet_10: functional & list comprehension
from functools import reduce

numbers = [1, 2, 3, 4, 5]
squares = [x*x for x in numbers]

def sum_of_squares(nums):
    return reduce(lambda a, b: a + b, (x*x for x in nums))
"""
]

# (3) Save snippets
for i, s in enumerate(snippets, 1):
    fname = f'snippets/snippet_{i:02d}.py'
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent(s).lstrip())
with open('snippets.json', 'w', encoding='utf-8') as f:
    json.dump(snippets, f, indent=2)

print(f"Saved {len(snippets)} snippets to 'snippets/' folder and to 'snippets.json'.")

# ==========================================================
# ---------- STEP 2 ----------
# AST parsing, feature extraction, tokenization (MiniLM, DistilRoBERTa, MPNet tokenizers)
# ==========================================================

import ast
from pathlib import Path
from typing import Dict, Any
from transformers import AutoTokenizer
import pprint

SNIPPET_DIR = Path('snippets')
OUTPUT_FILE = Path('processed_snippets.json')

TOKENIZER_SPECS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "distilroberta": "distilroberta-base",
    "mpnet": "sentence-transformers/all-mpnet-base-v2"
}

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.patterns = {k: False for k in [
            "uses_async","uses_decorator","uses_generator","uses_dataclass","uses_contextmanager",
            "uses_regex","uses_reduce","uses_listcomp","uses_fstring","uses_with",
            "uses_try","uses_lambda","uses_import_time"
        ]}

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        if node.decorator_list:
            self.patterns["uses_decorator"] = True
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for n in node.names:
            self.imports.append(n.name)

    def visit_ImportFrom(self, node):
        self.imports.append(node.module)

    def visit_AsyncFunctionDef(self, node):
        self.patterns["uses_async"] = True

def analyze_code(code: str) -> Dict[str, Any]:
    analyzer = CodeAnalyzer()
    try:
        tree = ast.parse(code)
    except Exception as e:
        return {"error": str(e)}
    analyzer.visit(tree)
    return {
        "functions": analyzer.functions,
        "classes": analyzer.classes,
        "imports": analyzer.imports,
        "patterns": analyzer.patterns,
        "ast_snippet": ast.dump(tree)[:400]
    }

print("Loading tokenizers...")
tokenizers = {tag: AutoTokenizer.from_pretrained(model) for tag, model in TOKENIZER_SPECS.items()}

processed = []
for p in sorted(SNIPPET_DIR.glob('snippet_*.py')):
    code = p.read_text(encoding='utf-8')
    meta = analyze_code(code)
    tokenization = {}
    for tag, tok in tokenizers.items():
        toks = tok(code, truncation=True, max_length=512)
        tokenization[tag] = {"token_count": len(toks['input_ids'])}
    processed.append({"filename": str(p), "analysis": meta, "tokenization": tokenization})

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(processed, f, indent=2)

print("\n Step 2 complete. Processed snippets and saved to 'processed_snippets.json'.")
pprint.pprint(processed[0])

# ==========================================================
# ---------- STEP 3 ----------
# Generate model explanation outputs
# ==========================================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

with open("processed_snippets.json", "r", encoding="utf-8") as f:
    processed = json.load(f)

codes = [s["filename"].split("/")[-1] for s in processed]
texts = [s["code"] for s in processed]

model_names = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "DistilRoBERTa": "sentence-transformers/all-distilroberta-v1",
    "MPNet": "sentence-transformers/all-mpnet-base-v2"
}

print("Loading models...")
models = {name: SentenceTransformer(m) for name, m in model_names.items()}

embeddings = {name: models[name].encode(texts, normalize_embeddings=True) for name in models}

concept_labels = [
    "code structure","algorithmic logic","mathematical reasoning","control flow",
    "function documentation","data handling","semantic meaning","variable naming clarity",
    "import organization","object-oriented design"
]

concept_embs = {m: models[m].encode(concept_labels, normalize_embeddings=True) for m in models}

explanation_outputs = []
for i, code in enumerate(texts):
    row = {"snippet": codes[i]}
    for model in models:
        sim = cosine_similarity([embeddings[model][i]], concept_embs[model])[0]
        top3 = np.argsort(sim)[::-1][:3]
        concepts = ", ".join([f"{concept_labels[j]} ({sim[j]:.2f})" for j in top3])
        row[f"{model}_output"] = f"{model} focuses on: {concepts}"
    explanation_outputs.append(row)

df = pd.DataFrame(explanation_outputs)
df.to_csv("model_explanations.csv", index=False)


# ==========================================================
# ---------- STEP 4 ----------
# Compare model explanation outputs and visualize
# ==========================================================

import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

df = pd.read_csv("model_explanations.csv")
models = ["MiniLM", "DistilRoBERTa", "MPNet"]

# Shared TF-IDF vocabulary
texts_all = sum([df[f"{m}_output"].tolist() for m in models], [])
vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(texts_all)
X = {m: vectorizer.transform(df[f"{m}_output"]) for m in models}

# Cross-model cosine similarity
def model_cosine(a,b): return np.mean(cosine_similarity(X[a], X[b]))
sim = pd.DataFrame([[model_cosine(a,b) for b in models] for a in models], index=models, columns=models)
print(sim)

# Dominant models
ranking = []
for _, row in df.iterrows():
    scores = {m: np.mean([float(x) for x in re.findall(r"\(([\d\.]+)\)", row[f"{m}_output"])]) for m in models}
    best = max(scores, key=scores.get)
    ranking.append({"snippet": row["snippet"], "best_model": best, **scores})
rank_df = pd.DataFrame(ranking)
print(rank_df)

# Visualizations
plt.figure(figsize=(7,5))
sns.heatmap(sim, annot=True, cmap="Blues")
plt.title("Model Explanation Similarity Heatmap")
plt.show()

plt.figure(figsize=(7,4))
rank_df["best_model"].value_counts().plot(kind="bar", color=["#8ecae6","#ffb703","#219ebc"])
plt.title("Dominant Model per Snippet")
plt.show()

# Word clouds
for m in models:
    txt = " ".join(df[f"{m}_output"])
    wc = WordCloud(width=600, height=400, background_color="white").generate(txt)
    plt.figure(figsize=(6,4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Focus Terms - {m}")
    plt.show()



