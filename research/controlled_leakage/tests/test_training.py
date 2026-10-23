import json
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from leaklab.backends import HFBackend
from leaklab.training import latest_adapter,sft


def test_real_lora_training_checkpoint_and_generation(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
    base = tmp_path/"base"
    base.mkdir()
    vocabulary = {s:i for i,s in enumerate(["<pad>","<s>","</s>","<unk>","<assistant>","prompt","response","one","two"])}
    tokenizer = Tokenizer(WordLevel(vocabulary,unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer,bos_token="<s>",eos_token="</s>",pad_token="<pad>",unk_token="<unk>")
    fast.chat_template = "{{ messages[0]['content'] }} <assistant>"
    fast.save_pretrained(base)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocabulary),hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,
        bos_token_id=1,eos_token_id=2,pad_token_id=0))
    model.save_pretrained(base)
    backend = HFBackend(str(base),"0"*40,trainable=True,device="cpu")
    checkpoint = tmp_path/"training"
    pairs = [("prompt one","response one"),("prompt two","response two")]
    state = sft(backend,pairs,checkpoint,epochs=1,seed=11)
    assert state["epoch"] == 1 and state["loss"] > 0
    assert any("lora_B" in name and p.abs().sum()>0 for name,p in backend.model.named_parameters())
    generated = backend.generate("prompt one",max_new_tokens=4,trace=True)
    assert generated.logprobs.shape[1] == len(vocabulary)
    assert generated.output_tokens == len(generated.token_ids)
    assert generated.logprobs.shape[0] == generated.output_tokens
    resumed = HFBackend(str(base),"0"*40,adapter=latest_adapter(checkpoint),trainable=True,device="cpu")
    assert sft(resumed,pairs,checkpoint,epochs=1,seed=11) == state
    with pytest.raises(ValueError,match="inputs changed"):
        sft(resumed,pairs+[("prompt","response")],checkpoint,epochs=1,seed=11)
