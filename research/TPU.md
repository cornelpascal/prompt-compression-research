# TPU execution (single device)

The research packages can select a PyTorch/XLA device explicitly. CPU and CUDA
remain the defaults. This is a **single-device portability path**, not a claim
that the full study fits a Colab TPU or that it has passed TPU hardware tests.

Use a Linux TPU runtime with matching PyTorch and `torch_xla` versions. For the
existing PyTorch 2.8 training environment, the upstream installation is:

```sh
python -m pip install -e 'research/prompt_compression[train,tpu,report,test]' \
  -e 'research/controlled_leakage[gpu,tpu,test]'
```

In Google Colab, select **v5e-1 TPU** and runtime version **2025.10** under
Runtime → Change runtime type. That image provides Python 3.12, PyTorch 2.8,
and PyTorch/XLA 2.8. The latest Colab image observed on 2026-11-01 provided
Python 3.13 and PyTorch 2.14, so installing this project's pinned TPU extra
there attempted to replace the preinstalled stack. On a 2025.10 v5e-1,
`promptcomp preflight --require-xla`, `promptcomp.attention_smoke`, and
`promptcomp.smoke` completed with tiny engineering fixtures. Run XLA commands
as separate processes; initializing the TPU in a notebook Python process and
then starting another TPU process can abort the second process.

Check the runtime's Python, PyTorch, and XLA versions before changing a Colab
installation. The project's CUDA Docker image and vLLM GPU server launcher are
not TPU images. Neither script should be used to install CUDA packages on a TPU.

## Verify the prompt-compression training path

Run these commands from the repository root in a TPU runtime. Use a fresh output
directory each time. The smoke uses tiny randomly initialized models and fixture
receivers, so its quality numbers are engineering checks only.

```sh
export PROMPTCOMP_DEVICE=xla
PYTHONPATH=research/prompt_compression python -m promptcomp preflight \
  --config research/prompt_compression/configs/attention.smoke.locked.json --require-xla
PYTHONPATH=research/prompt_compression python -m promptcomp.smoke \
  --output /content/promptcomp-tpu-smoke
PYTHONPATH=research/prompt_compression python -m promptcomp.attention_smoke \
  --output /content/promptcomp-attention-tpu-smoke
```

The attention scorer, rewrite backend, deletion backend, checkpoint path, and
optimizer now select XLA with `PROMPTCOMP_DEVICE=xla`; each optimizer update
issues an XLA execution boundary. `cloud/train.sh` accepts TPU mode with one
process. `cloud/run-study.sh` and `cloud/run-attention-study.sh` accept TPU mode
only when their teacher and receiver endpoints have already been supplied.
The supplied `cloud/serve.py` and vLLM image still serve CUDA models.

## Leakage package

`LEAKLAB_DEVICE=xla` selects XLA for the Hugging Face and inverter backends and
training steps. The finite-source theory and synthetic rule pilot remain CPU
work. The alternating neural trainer holds a compressor, task model, and attacker
at once; its own resource estimate is about 80 GB and has not been measured.
It **will not fit a 16 GB Colab v5e-1 chip as configured**. Model sharding or a
different sequential/offload algorithm would be needed before that experiment
can run there. No scientific TPU result is claimed.

## Limits

- TPU devices are not CUDA GPUs. Separate Colab notebooks cannot be combined
  into one tensor-parallel model or one distributed run by these launchers.
- XLA training here supports one TPU device per run. Multiple seeds may be run
  independently on separately granted runtimes, each with a distinct output path.
- The 4B rewrite model, 24B/32B frozen servers, and full leakage trainer need
  real memory measurements and suitable TPU serving/sharding work. A 16 GB v5e-1
  is intended for the tiny engineering smoke, not the full locked profiles.
- XLA compilation, model generation, LoRA adapter saving, and checkpoint resume
  must be verified on a real TPU. Local CPU checks cannot establish TPU support.
