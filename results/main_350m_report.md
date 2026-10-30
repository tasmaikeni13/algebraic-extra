# 350M Architecture Benchmark Report: Algebraic vs. Standard Transformer (7B Tokens)

**Generated:** 2026-10-30 10:05:53 IST  
**Scale:** 350M Active Parameters (`332,453,248` non-embedding weights, tied embeddings)  
**Dataset:** FineWeb-Edu (7.0 Billion Tokens per run, Chinchilla-optimal compute budget)  
**Hardware:** Cloud TPU v4-32 (16 TPUs across 4 workers: `w0`, `w1`, `w2`, `w3`)  
**Seed:** 42  

## Executive Summary

This benchmark compares the **Algebraic Transformer** against the modern **Standard LLaMA-style Transformer** at 350M scale trained on 7.0 Billion tokens of FineWeb-Edu using winning hyperparameter recipes discovered in the 350M sweep.

## Head-to-Head Benchmark Results

| Architecture | Optimizer | Held-Out NLL (nats) | Held-Out Perplexity | Final Train Loss | Delta NLL vs Standard | Relative PPL Reduction | Throughput | Runtime |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard LLaMA** | **muon_adamw** | 2.5036 | 12.23 | 2.5085 | baseline | baseline | 574k tok/s | 3.39 hrs |
| **Algebraic Transformer** | **muon_adamw** | **2.4909** | **12.07** | **2.4956** | **-0.0128** | **+1.27%** | 517k tok/s | 3.76 hrs |
| **Standard LLaMA** | **hybrid_muon_cadamw** | 2.5066 | 12.26 | 2.5118 | baseline | baseline | 574k tok/s | 3.39 hrs |
| **Algebraic Transformer** | **hybrid_muon_cadamw** | **2.4943** | **12.11** | **2.4994** | **-0.0123** | **+1.22%** | 517k tok/s | 3.76 hrs |
