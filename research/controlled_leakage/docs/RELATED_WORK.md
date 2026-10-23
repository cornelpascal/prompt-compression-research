# Closest work and contribution boundaries

Focused planning scan, checked 2026-10-17. This is an initial overlap map, not an exhaustive novelty review or evidence that local PaperKB records passed independent review. The local database was unavailable during implementation.

| Primary source | Relevant overlap | Boundary for this project |
|---|---|---|
| [Fundamental Limits of Prompt Compression](https://arxiv.org/abs/2407.15504) | Fixed-decoder prompt rate–distortion formulation and finite examples | Add an explicit sensitive variable and conditional leakage objective; do not claim the rate–distortion framing itself is new. |
| [From the Information Bottleneck to the Privacy Funnel](https://arxiv.org/abs/1402.1774) | Utility versus inference leakage in probabilistic data disclosure | Treat privacy–utility optimization as established; specify the fixed-decoder and token-budget specialization. |
| [LLMLingua-2](https://arxiv.org/abs/2403.12968) | Learned extractive compression | Use the actual pretrained compressor, with recorded revision and target-token counts, as a utility-oriented baseline. |
| [Understanding and Improving Information Preservation](https://aclanthology.org/2025.findings-emnlp.949/) | Compression evaluation includes grounding and preserved information | Distinguish desired fact preservation from recoverability of protected facts. |
| [Language Model Inversion](https://arxiv.org/abs/2311.13647) | Recovering input text from next-token probability information | Specify the observation surface and distinguish recovery of the actual compressed input from the pre-compression source. |
| [PILS](https://arxiv.org/abs/2506.17090) and [author code](https://github.com/dill-lab/PILS) | Sequence inversion using compact probability representations | The implementation adapts ALR sequence features and uses separate top-k encoders. Original-paper reproduction remains a separate validation task. |
| [PII-Bench](https://aclanthology.org/2026.acl-long.227/) | Query relevance of PII and masking evaluation | Requires explicit external-data adaptation; its standard labels cannot silently become downstream task answers. |
| [Adaptive Text Anonymization](https://aclanthology.org/2026.findings-acl.401/) and [author code](https://github.com/gabrielloiseau/adaptive-text-anonymization) | Reflective optimization of anonymization instructions for privacy and utility | Closest method overlap. Current baseline is independently implemented reflective search, not published GEPA reproduction. Investigate explicit achieved-length constraints and adaptive inversion as the distinguishing contribution. |
| [LLMs are Privacy Erasable](https://aclanthology.org/2025.findings-emnlp.304/) | Prompt privacy against reconstruction during inference | Compare access assumptions and architecture before making claims of superiority. |

The strongest candidate contribution is a reproducible comparison across explicit context/output disclosure channels, finite-source bounds, and learned compression under length and utility constraints. Neither literature novelty nor empirical improvement is established by implementing the package.
