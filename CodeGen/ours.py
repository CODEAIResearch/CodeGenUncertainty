import ast
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np

# AST-based phrase extraction for Python code
def extract_ast_phrases(code_snippet):
    phrases = []

    class CodeVisitor(ast.NodeVisitor):
        def generic_visit(self, node):
            if hasattr(node, 'lineno'):
                start_line = node.lineno
                end_line = getattr(node, 'end_lineno', start_line)
                lines = code_snippet.split('\n')[start_line-1:end_line]
                phrase = '\n'.join(lines).strip()
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
            super().generic_visit(node)

    tree = ast.parse(code_snippet)
    CodeVisitor().visit(tree)
    return phrases

# Load pretrained CodeBERT
tokenizer = AutoTokenizer.from_pretrained("/home/rsr200002/LLMs/codebertMLM")
model = AutoModelForMaskedLM.from_pretrained("/home/rsr200002/LLMs/codebertMLM")
model.eval()

# --- Mask phrases with individual [MASK] tokens ---
def mask_phrase_with_n_masks(code_snippet, phrase, tokenizer):
    tokenized_phrase = tokenizer.tokenize(phrase)
    mask_tokens = ' '.join([tokenizer.mask_token] * len(tokenized_phrase))
    return code_snippet.replace(phrase, mask_tokens)


# 4️⃣ Model confidence: average max token probability at masked positions
def compute_model_confidence1(code_snippet, tokenizer, model):
    inputs = tokenizer(code_snippet, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    input_ids = inputs.input_ids[0]
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return 1.0
    mask_probs = probs[0, mask_positions, :]
    max_probs, _ = mask_probs.max(dim=-1)
    confidence = max_probs.mean().item()
    return confidence


# 5️⃣ Compute importance scores (faithful to MARS)

def compute_model_confidence(code_snippet, tokenizer, model):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    input_ids = inputs.input_ids[0]
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return 1.0
    mask_probs = probs[0, mask_positions, :]
    max_probs, _ = mask_probs.max(dim=-1)
    confidence = max_probs.mean().item()
    return confidence


def compute_phrase_importance_scores(code_snippet, tokenizer, model):
    phrases = extract_ast_phrases(code_snippet)
    raw_scores = []
    token_counts = []

    original_confidence = compute_model_confidence(code_snippet, tokenizer, model)

    for phrase in phrases:
        masked_code = mask_phrase_with_n_masks(code_snippet, phrase, tokenizer)
        masked_confidence = compute_model_confidence(masked_code, tokenizer, model)
        o_k = masked_confidence / original_confidence
        importance = (1 - o_k)
        token_count = len(tokenizer.tokenize(phrase))
        raw_scores.extend([importance / token_count] * token_count)
        token_counts.append(token_count)

    raw_scores = np.array(raw_scores)
    tau = 0.01
    normalized_scores = np.exp(raw_scores / tau)
    normalized_scores /= normalized_scores.sum()

    phrase_scores = []
    idx = 0
    for count in token_counts:
        phrase_score = normalized_scores[idx:idx + count].sum()
        phrase_scores.append(phrase_score)
        idx += count

    # Semantic entropy as the final uncertainty score
    entropy = -np.sum(np.array(phrase_scores) * np.log(np.array(phrase_scores) + 1e-10))
    return entropy

def compute_phrase_importance_scores1(code_snippet):
    phrases = extract_ast_phrases(code_snippet)
    raw_scores = []
    token_counts = []

    original_confidence = compute_model_confidence(code_snippet, tokenizer, model)

    for phrase in phrases:
        masked_code = mask_phrase_with_n_masks(code_snippet, phrase, tokenizer)
        masked_confidence = compute_model_confidence(masked_code, tokenizer, model)
        o_k = masked_confidence / original_confidence
        importance = (1 - o_k)
        token_count = len(tokenizer.tokenize(phrase))
        raw_scores.extend([importance / token_count] * token_count)
        token_counts.append(token_count)

    # Softmax over all tokens
    raw_scores = np.array(raw_scores)
    tau = 0.01
    normalized_scores = np.exp(raw_scores / tau)
    normalized_scores /= normalized_scores.sum()

    # Aggregate per phrase
    phrase_scores = []
    idx = 0
    for count in token_counts:
        phrase_score = normalized_scores[idx:idx + count].sum()
        phrase_scores.append(phrase_score)
        idx += count

    return phrases, phrase_scores

def compute_semantic_entropy(phrase_scores):
    phrase_scores = np.array(phrase_scores)
    entropy = -np.sum(phrase_scores * np.log(phrase_scores + 1e-10))
    return entropy

import re
def remove_comments(code):
    # Remove single-line comments (#, //)
    code = re.sub(r"(#.*?$|//.*?$)", "", code, flags=re.MULTILINE)

    # Remove multi-line block comments /* ... */
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

    # Remove triple-quoted strings (Python docstrings)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

    return code

"""# 6️⃣ Example usage
code_snippet = 
from typing import List\ndef string_xor(a: str, b: str) -> str:\n    \"\"\" Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"\n    result = ''\n    for i in range(len(a)):\n        if a[i] == '1' and b[i] == '1':\n            result += '0'\n        elif a[i] == '0' and b[i] == '0':\n            result += '0'\n        else:\n            result += '1'\n    return result

code_snippet = remove_comments(code_snippet)
phrases, phrase_scores = compute_phrase_importance_scores(code_snippet)
semantic_entropy = compute_semantic_entropy(phrase_scores)

# Output
for phrase, score in zip(phrases, phrase_scores):
    print(f"Phrase: {phrase} | Importance: {round(score, 4)}")

print(f"\nSemantic Entropy: {round(semantic_entropy, 4)}")"""


filename = "/home/rsr200002/CodeGen/evalplus_results/humaneval/home--rsr200002--LLMs--CodeLlama_7b_Instruct_hf_hf_temp_0.0.eval_results.json"

import json
with open(filename, 'r') as f:
    data = json.load(f)

scores = []
labels = []

for task_group in data['eval'].values():
    for task in task_group:
        solution = task['solution']
        status = task['plus_status']
        label = 1 if status == 'fail' else 0
        score = compute_phrase_importance_scores(solution, tokenizer, model)
        scores.append(score)
        labels.append(label)

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labels, scores)
print(f"AUC: {auc:.4f}")


