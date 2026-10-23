"""Small randomly initialized models validate neural paths without downloads."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from leaklab.inversion import SequenceInverter
from leaklab.backends import HFBackend


@pytest.mark.parametrize("access",["full","top5","top20"])
def test_probability_inverter_has_real_gradients_and_generation(access):
    from transformers import T5Config,T5ForConditionalGeneration
    torch.manual_seed(1)
    decoder = T5ForConditionalGeneration(T5Config(vocab_size=32,d_model=16,d_ff=32,num_layers=1,
        num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1))
    net = SequenceInverter(decoder,access,24,feature_dim=6)
    rng = np.random.default_rng(1)
    k = 5 if access == "top5" else 20
    data = {"features":rng.normal(size=(3,6)).astype(np.float32)} if access == "full" else {
        "token_ids":np.arange(k)[None,:].repeat(3,axis=0),"logprobs":np.full((3,k),-4.,dtype=np.float32)}
    public = torch.tensor([[3,4,5,1]])
    labels = torch.tensor([[6,7,1]])
    loss = net(data,public,labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert net.project[0].weight.grad.abs().sum() > 0
    net.eval()
    assert net.reconstruct(data,public,max_new_tokens=3).shape[0] == 1


def test_response_only_likelihood_uses_exact_sampled_ids():
    from transformers import GPT2Config,GPT2LMHeadModel
    backend = HFBackend.__new__(HFBackend)
    backend.model = GPT2LMHeadModel(GPT2Config(vocab_size=32,n_layer=1,n_head=2,n_embd=16,n_positions=32))
    backend.model.eval()
    backend.device = torch.device("cpu")
    backend.prompt_ids = lambda prompt:[2,3,4]
    tokens = [7,8,1]
    actual = backend.response_ids_logprob("ignored",tokens)
    logits = backend.model(input_ids=torch.tensor([[2,3,4,7,8,1]])).logits[0,2:5].log_softmax(-1)
    expected = logits[torch.arange(3),torch.tensor(tokens)].sum()
    assert torch.allclose(actual,expected)
    actual.backward()
    assert any(p.grad is not None for p in backend.model.parameters())


def test_public_probability_prior_has_no_probability_positions():
    from transformers import T5Config,T5ForConditionalGeneration
    decoder = T5ForConditionalGeneration(T5Config(vocab_size=16,d_model=8,d_ff=16,num_layers=1,
        num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,pad_token_id=0))
    net = SequenceInverter(decoder,"full",16,feature_dim=4)
    public = torch.tensor([[2,3,4]])
    assert net.inputs(None,public)["inputs_embeds"].shape == (1,3,8)


def test_rank_diagnostic_identifies_lossy_coordinates():
    from leaklab.observations import alr_rank_diagnostic
    head = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[2.,0.]])
    assert alr_rank_diagnostic(head,[0,1,2])["sufficient_for_linear_head"]
    assert not alr_rank_diagnostic(head,[0,1,3])["sufficient_for_linear_head"]


def test_full_neural_observation_training_and_evaluation_pipeline(tmp_path):
    import json
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast,
        T5Config,T5ForConditionalGeneration)
    from leaklab.data import generate_dataset
    from leaklab.io import atomic_json,read_jsonl
    from leaklab.runner import run,export_attack_data
    from leaklab.schema import ExperimentConfig
    from leaklab.inversion import train_inverter,TrainedInverter
    from leaklab.evaluation import evaluate_attacker

    vocabulary = {s:i for i,s in enumerate(["<pad>","</s>","<unk>"]+[f"word{i}" for i in range(29)])}
    token = Tokenizer(WordLevel(vocabulary,unk_token="<unk>"))
    token.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=token,eos_token="</s>",pad_token="<pad>",unk_token="<unk>")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    causal,decoder = tmp_path/"causal",tmp_path/"decoder"
    for folder in (causal,decoder):
        tokenizer.save_pretrained(folder)
    LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
        max_position_embeddings=512,eos_token_id=1,pad_token_id=0)).save_pretrained(causal)
    T5ForConditionalGeneration(T5Config(vocab_size=32,d_model=16,d_ff=32,num_layers=1,
        num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1)).save_pretrained(decoder)
    revision = "0"*40
    config = ExperimentConfig(backend="hf",seeds=(11,),ratios=(.5,),methods=("empty",),
        task_models=(str(causal),),revisions={str(causal):revision},max_new_tokens=2,
        probability_access=("text","top5","top20","full"),bootstrap_samples=100)
    config_path = tmp_path/"config.json"
    atomic_json(config_path,config.model_dump())
    data,run_dir = tmp_path/"data",tmp_path/"run"
    generate_dataset(data,size=6)
    run(config_path,data,run_dir)
    export = tmp_path/"export.jsonl"
    export_attack_data(run_dir,export)
    row = next(r for r in read_jsonl(export) if r["observation"]["access"]=="full")
    checkpoint,prior = tmp_path/"attack",tmp_path/"prior"
    options = dict(export=export,run_directory=run_dir,scope=row["scope"],model=str(decoder),revision=revision)
    train_inverter(**options,output=checkpoint)
    saved_time = (checkpoint/"epoch-001.pt").stat().st_mtime_ns
    train_inverter(**options,output=checkpoint)
    assert (checkpoint/"epoch-001.pt").stat().st_mtime_ns == saved_time
    train_inverter(**options,output=prior,public_only=True)
    trained = TrainedInverter(checkpoint)
    assert trained.metadata["training_ids"] == [r["id"] for r in read_jsonl(export) if r["scope"]==row["scope"]]
    result = evaluate_attacker(run_dir,data/"test.jsonl",checkpoint,tmp_path/"evaluation",prior)
    assert result["rows"] == 12 and result["reconstruction_rows"] == 0
    assert json.loads((tmp_path/"evaluation"/"manifest.json").read_text())["public_prior"] == "independent_neural_public"
    times = {p.name:p.stat().st_mtime_ns for p in (tmp_path/"evaluation"/"cases").glob("*.json")}
    assert evaluate_attacker(run_dir,data/"test.jsonl",checkpoint,tmp_path/"evaluation",prior) == result
    assert times == {p.name:p.stat().st_mtime_ns for p in (tmp_path/"evaluation"/"cases").glob("*.json")}
    with pytest.raises(ValueError,match="frozen dataset"):
        evaluate_attacker(run_dir,data/"shift.jsonl",checkpoint,tmp_path/"wrong")
