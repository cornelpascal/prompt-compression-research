# Finite-source theory and numerical certificates

Let X be a finite source including the context and known query, S=s(X) the protected attribute, and W=w(X) public information. W must include the query, policy, decoder identity, and any auxiliary knowledge included in the threat model. The compressor is a stochastic channel C(z|x) over an explicitly enumerated message alphabet. A fixed decoder induces a precomputed nonnegative task distortion d(x,z). Each message has length l(z).

The implemented optimization minimizes I(S;Z|W) subject to:

```
C(z|x) >= 0; sum_z C(z|x) = 1
sum_xz p(x) C(z|x) l(z) <= R
sum_xz p(x) C(z|x) d(x,z) <= D
```

The source distribution, message alphabet, public partition, lengths, and decoder distortion table are fixed. Writing t(s,w,z)=sum_{x:s(x)=s,w(x)=w} p(x)C(z|x), the objective in bits is:

```
sum_swz t(s,w,z) log2( t(s,w,z) p(w) / (p(s,w) p(w,z)) )
```

Conditional mutual information is convex in this channel for the fixed source. The constraints are linear. This differs from privacy-funnel formulations that constrain utility using a mutual-information superlevel set, which can be nonconvex. It also differs from neural training, where parameterization and task rewards are nonconvex.

## Numerical method

Before optimization, symbols with identical distortion columns are merged into the shortest representative. This preserves task error, cannot increase expected length, and cannot increase leakage by data processing. Every reduced channel remains feasible in the original alphabet, so the two optimum values coincide. The returned channel is expanded to the original alphabet. Removing these redundant symbols also tightens the numerical supporting-plane certificate.

1. A linear program checks feasibility. A solver failure is distinguished from proven LP infeasibility.
2. Feasible LP vertices construct a relative-interior starting point.
3. SLSQP optimizes a slightly smoothed objective; feasibility and the final unsmoothed leakage are evaluated separately.
4. At a strictly positive channel C0, convexity gives `f(C) >= f(C0) + grad(f(C0)) · (C-C0)`.
5. Minimizing that supporting plane over the original feasible polytope gives a lower bound. The feasible candidate provides an upper bound. Their difference is reported.

`numerically_certified` means the residual and supporting-plane gap meet numerical tolerances. It is not an interval-arithmetic proof. `feasible_gap_remaining` preserves the candidate and its unresolved gap instead of claiming optimality. Certificates apply only to the enumerated source and fixed decoder. The expected-length constraint is a relaxation of the per-example hard budgets used by the neural experiment.

## Binary controls

T is a uniform useful bit, S is a protected bit, and the decoder outputs T from messages `empty`, `0`, or `1`. Their lengths are 0, 1, 1, and the decoder outputs 0, 0, 1 respectively. Message boundaries are observed: an empty message can be informative. Expected token length here is not a bit-rate or a prefix-code length.

- Independent: S is independent of T. Zero task error is possible with zero leakage once the budget permits transmitting T.
- Correlated: P(S=T)=0.9. At zero task error the minimum leakage is I(S;T)=1-h2(0.1), approximately 0.531 bits.
- Essential: S=T. Zero task error requires one bit of leakage. With binary uniform S, a decoder error at most D<=1/2 implies I(S;Z)>=1-h2(D), a direct application of Fano's inequality. This standard consequence is a validation target, not a claimed novel theorem.

In all three controls, exact task prediction requires expected length at least 0.5 under this particular decoder. The tests verify these boundaries, conditional leakage when W already reveals S, gradients, and data processing.

## Output disclosure

If the frozen model and its generation randomness access X only through Z and W, then S → (Z,W) → O is a Markov chain and:

```
I(S;O|W) <= I(S;Z|W)
```

O may include generated text and disclosed token probabilities. This inequality requires matched public information and no additional access to the original context. An output can nevertheless make a finite attacker's inference easier, so empirical recovery rates need not respect the same ordering across different attack algorithms.

Successfully reconstructing S demonstrates leakage under that attack. Failing to reconstruct S does not upper-bound information leakage. Random independent secrets and held-out attacker families address some practical failure modes without establishing differential privacy or cryptographic secrecy.

## Foundations and open research

The formulation builds on [fixed-decoder prompt rate–distortion](https://arxiv.org/abs/2407.15504) and [the privacy funnel](https://arxiv.org/abs/1402.1774). Extending the finite model to larger task families, deriving tighter analytical frontiers, and connecting certified finite limits to learned natural-language compressors remain research work. The implementation supplies a solver and falsifiable controls; it does not assert a new general theorem.
