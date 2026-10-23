export const methods = [
  {
    id: "rewrite",
    number: "01",
    group: "Prompt compression",
    name: "Rewrite the prompt",
    subtitle: "Generative compression",
    description:
      "A language model turns the instruction, passage, and question into shorter ordinary text. The receiver gets the compressed prompt alone.",
    formula:
      "reward = min(F1_Qwen, F1_Mistral) − 2 × max(0, length / budget − 1)",
    steps: [
      "Generate four teacher candidates per prompt and budget.",
      "Rank valid candidates by the weaker frozen receiver’s F1.",
      "Warm-start the rewriter with supervised response likelihood.",
      "Refine using group-relative policy updates and a frozen reference.",
    ],
    inputs:
      "Public prompt and token budget. Gold answers are isolated in the offline scorer.",
    outputs:
      "A shorter text prompt, token counts, elapsed time, and explicit budget or generation-limit status.",
    model: "Qwen3-4B-Instruct-2507 with LoRA; Qwen3-32B teacher.",
    why: "Rewriting can merge redundant statements and express the same task more concisely.",
    limits:
      "It can invent facts, drop constraints, or forward an answer instead of preserving a task. Lexical diagnostics cannot prove semantic fidelity.",
    tests:
      "Response-only likelihood, exact sampled-token scoring, real LoRA updates, overflow accounting, and interrupted-run recovery.",
    source: "research/prompt_compression/promptcomp/training.py",
    command:
      "promptcomp train --method rewrite --stage sft --config CONFIG --data DATA --supervision SUPERVISION --seed 11 --output RUN",
  },
  {
    id: "deletion",
    number: "02",
    group: "Prompt compression",
    name: "Select source words",
    subtitle: "Extractive compression",
    description:
      "A ModernBERT encoder assigns a keep/drop score to each source word. Selected words retain their original wording and order.",
    formula: "P(mask | prompt, budget) = ∏ Bernoulli(keepᵢ ; sigmoid(logitᵢ))",
    steps: [
      "Align teacher extractions to original word spans.",
      "Train word-level keep/drop scores with binary cross-entropy.",
      "Optimize sampled masks using a leave-one-out reward baseline.",
      "Remove low-scoring words until the measured text fits the budget.",
    ],
    inputs: "The complete public prompt and a reference-token budget.",
    outputs:
      "Compressed text plus source character spans for every retained word.",
    model: "ModernBERT-base with a word-level linear head.",
    why: "Source spans make deletions auditable and prevent the selector from inventing new words.",
    limits:
      "Deleting a negation or qualifier can change the meaning. Exact wording alone does not guarantee task preservation. This method runs the full encoder.",
    tests:
      "Subsequence alignment, word-to-token mapping, source order, Bernoulli gradients, budget projection, and checkpoint recovery.",
    source: "research/prompt_compression/promptcomp/compressors.py",
    command:
      "promptcomp train --method delete --stage sft --config CONFIG --data DATA --supervision SUPERVISION --seed 11 --output RUN",
  },
  {
    id: "attention",
    number: "03",
    group: "Prompt compression",
    name: "Score with attention",
    subtitle: "First-layer Q/K only",
    description:
      "A lightweight scorer reads every candidate token but runs only the computations needed for first-layer attention. It compares frozen weights with trained query/key projections.",
    formula: "A = softmax(QKᵀ / √d + causal mask)",
    steps: [
      "Embed the prompt and append a scoring-only copy of the question.",
      "Apply input normalization, Q/K projections, Q/K normalization, and rotary positions.",
      "Compute globally normalized causal attention in blocks.",
      "Average across query positions, heads, and subtokens; select source words.",
    ],
    inputs:
      "Public prompt, budget, and a character range identifying the question. The question copy is never sent separately to the receiver.",
    outputs:
      "Source-word selection and ordinary text. There is no value projection, feed-forward block, later layer, or answer generation.",
    model: "Selected first-layer tensors from the pinned Qwen3-4B checkpoint.",
    why: "The experiment tests whether useful token rankings can be obtained without a full language-model pass.",
    limits:
      "Attention is not a correctness probability. First-layer token representations lack deep contextual reasoning. The 98% F1 criterion must be tested on held-out data.",
    tests:
      "Full-layer score equivalence, dense versus blockwise gradients, Q/K-only updates, Unicode spans, and calibration identity checks.",
    source: "research/prompt_compression/ATTENTION.md",
    command:
      "promptcomp train --method attention --stage sft --config CONFIG --data DATA --supervision SUPERVISION --seed 11 --output RUN",
  },
  {
    id: "privacy",
    number: "04",
    group: "Controlled leakage",
    name: "Compress with privacy constraints",
    subtitle: "Utility versus recoverability",
    description:
      "Synthetic records make task-relevant information and protected values explicit. The research measures how compression changes both task success and the ability of an attacker to recover private information.",
    formula:
      "objective = task utility − privacy penalty − budget overflow penalty",
    steps: [
      "Generate disjoint entity/template splits and private-code counterfactuals.",
      "Compare original, empty, redaction, and redaction-plus-compression controls.",
      "Train a compressor and adaptive attacks on training observations only.",
      "Evaluate actual recovery, utility, length, and correlation shift.",
    ],
    inputs:
      "Public task, structured source record, budget, and a declared observer access level.",
    outputs:
      "Compressed observations and utility/recovery measurements for each threat model.",
    model:
      "CPU pilot: deterministic rule task. Neural trainers and pinned GPU configurations are implemented separately.",
    why: "A shorter prompt may retain task information while also revealing a secret through correlation or side channels.",
    limits:
      "The CPU controls are not a general natural-language PII detector. Lower recovery is meaningful only alongside comparable task utility and achieved length.",
    tests:
      "Counterfactual answer invariance, disjoint splits, observation boundaries, literal recovery, and budget compliance.",
    source: "research/controlled_leakage/docs/PROTOCOL.md",
    command:
      "leaklab run --config research/controlled_leakage/configs/pilot.json --data DATA --output RUN",
  },
  {
    id: "attacks",
    number: "05",
    group: "Controlled leakage",
    name: "Measure reconstruction attacks",
    subtitle: "Explicit observer capabilities",
    description:
      "Attackers receive only the declared observation: compressed context, released text, public information, or configured probability traces. Separate scopes prevent results from different access levels being conflated.",
    formula: "excess recovery = attacker recovery − public-only prior recovery",
    steps: [
      "Export training observations without evaluation identities.",
      "Train a separate attacker for each model/method/budget/access scope.",
      "Evaluate literal protected-value recovery and reconstruction measures.",
      "Compare against a matched public-only prior and report uncertainty.",
    ],
    inputs:
      "Declared observations and training labels; evaluation labels stay outside attacker inputs.",
    outputs:
      "Recovery rates, reconstruction metrics, and paired comparisons with explicit access assumptions.",
    model:
      "Copying and attribute attacks in the CPU pilot; neural inversion is tested with tiny random models.",
    why: "A compressed string can look anonymous while still allowing inference of protected information.",
    limits:
      "Failure of one attacker does not establish privacy against all attackers. Probability-trace attacks require the corresponding access.",
    tests:
      "Training/evaluation separation, probability-trace shape, inverter training/reload, and scoped result identities.",
    source: "research/controlled_leakage/leaklab/inversion.py",
    command: "leaklab list-scopes --export RUN/attack-training.jsonl",
  },
  {
    id: "theory",
    number: "06",
    group: "Controlled leakage",
    name: "Solve a finite privacy frontier",
    subtitle: "A controlled information-theoretic model",
    description:
      "A finite source and fixed decoder make a small privacy–utility optimization problem numerically inspectable. Independent, correlated, and task-essential secrets behave differently.",
    formula:
      "minimize I(secret ; released symbol), subject to length and task-error limits",
    steps: [
      "Define the finite joint distribution and decoder.",
      "Choose expected-length and task-error budgets.",
      "Optimize the release channel and compute a lower bound.",
      "Inspect feasibility, residuals, and the numerical optimality gap.",
    ],
    inputs: "A finite regime, expected-length budget, and task-error budget.",
    outputs:
      "Feasible or infeasible status, leakage in bits, a release channel, and a floating-point certificate.",
    model: "Finite-source optimization; no language model is involved.",
    why: "These controls isolate when useful information inherently reveals protected information.",
    limits:
      "A floating-point certificate for a fixed finite problem is not a general privacy guarantee for language models.",
    tests:
      "Known binary controls, feasibility, constraint residuals, and bound-gap checks.",
    source: "research/controlled_leakage/docs/THEORY.md",
    command: "leaklab theory --output frontier.json",
  },
];

export const testGroups = [
  {
    name: "Numerical attention correctness",
    project: "Compression",
    checks:
      "Reduced first-layer attention versus the full layer; blockwise versus dense gradients.",
    meaning:
      "Establishes that the cheaper computation implements the intended scores. It does not establish useful token rankings.",
    source: "research/prompt_compression/tests/test_attention.py",
  },
  {
    name: "Training and recovery",
    project: "Compression",
    checks:
      "Real parameter updates, frozen non-Q/K parameters, optimizer restoration, and two-rank synchronization.",
    meaning:
      "Confirms reproducible execution and checkpoint integrity on CPU. CUDA/NCCL still needs validation.",
    source: "research/prompt_compression/tests/test_training.py",
  },
  {
    name: "Calibration and held-out evaluation",
    project: "Compression",
    checks:
      "Non-monotonic quality curves, missing pairs, zero baselines, no feasible setting, identity mismatch, and transfer failure.",
    meaning:
      "Prevents test-set tuning and prevents unsuccessful compression from being presented as qualified.",
    source: "research/prompt_compression/tests/test_attention_calibration.py",
  },
  {
    name: "Evidence and source boundaries",
    project: "PaperKB",
    checks:
      "Independent review identities, source-version chronology, coverage, immutable submissions, and graph barriers.",
    meaning:
      "A citation candidate is not automatically ancestry; incomplete sources cannot pass production publication.",
    source: "tests/test_source_versions.py",
  },
  {
    name: "Synthetic privacy controls",
    project: "Controlled leakage",
    checks:
      "Disjoint entities and templates, counterfactuals, task invariance, bounded outputs, and literal secret recovery.",
    meaning:
      "Checks the controlled experiment and threat-model boundaries; it is not a broad privacy certification.",
    source: "research/controlled_leakage/tests/test_pipeline.py",
  },
  {
    name: "Neural inversion and finite theory",
    project: "Controlled leakage",
    checks:
      "Probability observations, inverter train/reload/resume, LoRA updates, and finite optimization controls.",
    meaning:
      "Establishes functioning research components using tiny models and a bounded mathematical problem.",
    source: "research/controlled_leakage/tests/test_neural.py",
  },
];

export const metrics = [
  [
    "F1",
    "Token overlap between a predicted answer and the best matching SQuAD reference. Reported on a 0–1 scale; the UI displays percentages.",
  ],
  [
    "Exact match",
    "Whether a normalized prediction exactly matches a gold answer. It is stricter than partial-overlap F1.",
  ],
  [
    "Budget-compliant F1",
    "The answer F1, with invalid-budget outputs counted as zero. This is the score used for attention calibration.",
  ],
  [
    "Relative F1",
    "Compressed F1 divided by the same receiver’s original-prompt F1. The acceptance target is at least 0.98 for both calibration receivers.",
  ],
  [
    "Achieved retention",
    "Actual output length divided by original length. Reference tokens, receiver chat-template tokens, and UTF-8 bytes are distinct units.",
  ],
  [
    "Secret recovery",
    "Literal protected-value recovery under the declared attacker and observation access. Lower values alone do not prove a better privacy–utility tradeoff.",
  ],
  [
    "Confidence interval",
    "A cluster bootstrap interval: articles for SQuAD, entities for the synthetic study. Small fixture samples are not evidence of real-model reliability.",
  ],
  [
    "Leakage in bits",
    "Mutual information in the finite-source model. It is not the same metric as empirical secret recovery.",
  ],
];
