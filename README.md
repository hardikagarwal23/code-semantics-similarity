# Code Semantic Similarity using SmolLM-1.7B + QLoRA

Fine-tuned **SmolLM-1.7B** using **4-bit QLoRA** to determine whether two code snippets are semantically equivalent.

## Problem

**Task:** Your goal is to determine if two provided snippets of code are semantically equivalent. Semantic equivalence means they achieve the same result or compute the same logic, even if the syntax or approach differs.
- `True` → Both snippets are semantically equivalent
- `False` → They implement different logic

**Objective:** Push the boundaries of what small, open-weight models can achieve in code understanding tasks without relying on massive compute or parameter counts.

---



## Approach

I fine-tuned **SmolLM-1.7B** as a binary sequence classifier.

Achieved an **F1 score of 0.93826 on the final hidden test set** in the IITG.ai Code Semantics Similarity Challenge.

### Pipeline

```text
Code 1 + Code 2
       ↓
Tokenization
       ↓
Maximum length = 512
       ↓
SmolLM-1.7B
       ↓
4-bit Quantization
       ↓
QLoRA / LoRA
       ↓
Binary Classification
       ↓
True / False
```

### Token Length Analysis

Token lengths were analyzed on a random sample of **50,000 code pairs** using the SmolLM tokenizer.

<table>
  <tr>
    <th>Original Code</th>
    <th>After Basic Cleaning</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/bc272c2c-1037-41ac-be35-a68ff4dcb203" width="400"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/97b86472-b259-4686-8e0d-c1d8fcadbbe0" width="400"/>
    </td>
  </tr>
</table>

Basic cleaning improved the ≤512-token coverage by only **0.65 percentage** points (73.09% → 73.74%), so the current pipeline uses the original code without regex-based cleaning.


### Possible Improvements
- **Balanced Token Allocation:** Dynamically divide the 512-token budget between `func1` and `func2` instead of truncating only at the end.
- **Preserve Important Code:** Keep a small portion from both the beginning and end of each function to retain declarations, core logic, and return/output statements.
