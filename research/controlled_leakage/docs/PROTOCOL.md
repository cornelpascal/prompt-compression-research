# Experimental protocol

## Fixed experimental boundaries

English, single-turn, text compression. Compressors see source context, query, protected attribute names and a token budget. Task models remain frozen. Gold task answers and protected values are available to supervised training and trusted scoring only. Context observers see compressed context; output observers see generated text and exactly their permitted probability channel. Both see the same public query, attribute names and auxiliary fields.

The training artifact deliberately contains observations and labels in separate fields. Passing an entire labelled row to a backend is a protocol violation. The API's strict Observation schema rejects extra keys. Scientific text and dataset contents are untrusted evidence, not instructions to the experiment controller.

Synthetic dataset seeds, split/index identifiers, generator manifests and held-out labels are evaluator metadata, not attacker side information. Opaque customer names prevent exposing a split, seed and row index inside the context. Reproducible seeded strings are not cryptographic secrets: disclosure of the generation seed and generator can allow corpus regeneration. The threat model withholds these until evaluation is complete.

## Before main-test execution

1. Check the related-work matrix and verify full methods of the closest baselines. Record differences from any independently implemented adaptation.
2. Run the CPU and tiny-model tests, then a GPU smoke run on a separate small dataset. Confirm generation traces, memory use, tokenizer counts, checkpoint restoration, and pinned revisions.
3. Use training/validation data to set sample size, training epochs, reward weights, and the primary model/budget/domain cells. Save the frozen configuration before inspecting test results.
4. Independently train compressor seeds 11, 23 and 47. A repeated deterministic baseline is not three trained models.
5. Select adaptive-anonymization instructions on validation only. Keep its search compute separate from inference costs.

## Main measurements

Secret recovery and compressed-input exact reconstruction use literal string equality, preserving case and punctuation. Task-essential private-code answers also require literal equality. Ordinary task-answer accuracy ignores case and punctuation; required-fact recall uses normalized lexical matching.

Report utility, achieved context length, budget violations, and protected-attribute recovery jointly, with exact recovery for random private codes and attribute accuracy for the finite group label. Report the full vector of metrics; averaging secret types can conceal easy attribute leakage and unrecoverable random strings.

Report original-context, empty-context, structured redaction, redaction plus compression, LLMLingua-2, plain summarization, tuned anonymization, and learned-compressor results. Retrain attacks for each compressor, budget, model, and seed. Online training adversaries are distinct from final independently fitted attackers. Reserve Mistral text attackers for final evaluation when Qwen attackers trained the compressor.

For probability observations, train separate top-5, top-20 and ALR/full-access inverters. Train matched public-only priors. Keep compressed-input reconstruction and original-private-fact reconstruction in separate checkpoints and tables. Inspect the output-head ALR rank/condition diagnostic before interpreting the full-access representation.

## Robustness and ablations

- Test and training templates/entities are disjoint; paired variants change private_code only.
- The `shift` split reverses the group–task correlation from 0.9 to 0.1. Report it separately from the ordinary test split.
- Evaluate both frozen receiver families using target-specific token counts.
- Run no-privacy-objective, frozen-online-attacker, and no-compressor-query ablations with the same source splits and evaluation cases.
- Check utility and leakage for essential secrets separately. A method that hides a required answer by failing the task has not solved the privacy conflict.

## Statistical interpretation

Use entity-paired differences and cluster bootstrap intervals: the unit includes both counterfactual variants and all training seeds. Report seed means separately; three seeds give limited information about training variance. Intervals across the full matrix are exploratory unless a primary family and multiplicity policy were preregistered.

The comparison command groups by target model, requested ratio, observer, attacker, domain and regime. It reports differences in recovery, task accuracy and achieved token count. A suggested eligibility check is mean achieved length within 5% and utility within two percentage points. These thresholds describe matching, not a universal success standard. Choose a primary baseline using validation; selecting the weakest baseline after inspecting test results is invalid.

The method hypothesis requires an improvement over a strong eligible baseline that reproduces across receiver families and survives independently retrained attackers. A well-supported null result still supports the benchmark and finite-model analysis, but refutes the corresponding improvement claim.

## External data and provenance

PII-Bench's PII-relevance/masking annotations are not automatically answers to its underlying queries. Import separately annotated task answers and protected-value dictionaries; retain the source URL, raw-file hash, mapping, and output hash. Do not train or select policies on this external evaluation set. No private organizational records are required.

PaperKB supplies the bibliography. This package never writes its database, changes extraction/review state, assigns graph parents, or bypasses its pilot/global barriers. PaperKB's independent paper-review workflow remains separate from software tests and this experimental protocol.
