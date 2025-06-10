
<p align="center">
  <a href="https://arxiv.org/abs/2506.00789" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/arXiv-2506.00789-B31B1B?style=flat-square&logo=arxiv" alt="arXiv">
</a>
<a href="https://huggingface.co/datasets/Rabinovich/RARE" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/HuggingFace-Dataset-%23FFD21E?style=flat-square&logo=huggingface"
         alt="HF dataset"></a>
</p>

> ***RARE*** is a unified framework designed to automatically generate synthetic, dynamic, and time-sensitive corpora for testing Retrieval-Augmented Generation (RAG) systems using domain-specific unstructured datasets. It also provides a benchmark that thoroughly evaluates the robustness of RAG systems under various perturbations.

## ✨  Key Features

- ***RARE*-Get**: a novel dynamic synthesis pipeline that automatically constructs time-sensitive evaluation data through knowledge graph triplet extraction and traversal techniques, enabling the creation of single-hop and multi-hop tuples (question, answer, ground truth chunks) at various complexity levels without manual curation from unstructured dataset.
- ***RARE*-Set**: a large-scale benchmark comprising over 400 specialized documents and 48,322 queries across financial, economics, and policy domains
- ***RARE*-Met**: a comprehensive robustness evaluation metric for measuring RAG system performance under perturbations to queries, documents, and simulated real-world retrieval results.

## 📦  Installation
