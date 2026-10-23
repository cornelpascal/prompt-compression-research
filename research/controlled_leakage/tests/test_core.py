import json
import math
import numpy as np
import pytest

from leaklab.compressors import ByteTokenizer, Compressor, bounded_text
from leaklab.data import generate_dataset, generate_split, load_examples, validate_splits
from leaklab.metrics import cluster_interval, paired_difference
from leaklab.observations import observe, probability_view, alr_features
from leaklab.schema import ExperimentConfig, Observation
from leaklab.theory import binary_problem, leakage, output_leakage, solve
from leaklab.training import require_train, reward_components


def test_generated_splits_counterfactuals_and_checksums(tmp_path):
    generate_dataset(tmp_path, size=12)
    splits = [load_examples(tmp_path/f"{s}.jsonl") for s in ("train","validation","test","shift")]
    validate_splits(sum(splits, []))
    assert splits[0] == generate_split("train", 12)
    for rows in splits:
        for a,b in zip(rows[::2],rows[1::2]):
            assert a.entity_id not in a.request.context
            assert a.pair_id == b.pair_id
            assert a.secrets["private_code"] != b.secrets["private_code"]
            if a.regime != "essential":
                assert a.answer == b.answer
    with (tmp_path/"test.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load_examples(tmp_path/"test.jsonl")


def test_train_guard_and_group_split():
    examples = generate_split("test", 2)
    with pytest.raises(ValueError,match="Training"):
        require_train(examples)
    with pytest.raises(ValueError,match="crosses"):
        validate_splits([examples[0],examples[1].model_copy(update={"split":"train"})])


def test_observations_exclude_labels_and_original_context():
    e = generate_split("test", 1)[0]
    for access in ("public","text","top5","top20","full"):
        obs = observe(e.request, access, "DO_NOT_EXPOSE", "visible answer")
        value = obs.model_dump_json()
        assert e.answer not in value or e.answer in e.request.query  # query is intentionally public
        assert e.secrets["private_code"] not in value
        assert "DO_NOT_EXPOSE" not in value
        assert e.request.context not in value
        assert "split" not in Observation.model_fields
    with pytest.raises(TypeError):
        observe(e,"context")
    with pytest.raises(ValueError):
        Observation(access="text",query="",public_auxiliary={},protected_attributes=(),secrets={"x":"y"})


def test_budget_unicode_and_unconstrained_controls():
    e = generate_split("test",1)[0]
    tok = ByteTokenizer()
    for budget in range(20):
        assert len(tok.encode(bounded_text("こんにちは café \n hello",budget,tok))) <= budget
        out = Compressor("redact_compress",tok).compress(e.request.context,e.request.query,e.request.privacy_policy,budget)
        assert out.token_count <= budget
    assert Compressor("original",tok).compress(e.request.context,e.request.query,e.request.privacy_policy,1).token_count > 1


def test_probability_views_do_not_renormalize_or_impute():
    p = np.arange(1.,25.).reshape(2,12)
    p /= p.sum(1,keepdims=True)
    logs = np.log(p)
    top = probability_view(logs,"top5")
    assert top.logprobs.shape == (2,5)
    assert np.all(np.exp(top.logprobs).sum(1) < 1)
    assert np.allclose(top.logprobs,np.take_along_axis(logs,top.token_ids,1))
    with pytest.raises(ValueError,match="full access"):
        alr_features(top,[0,1,2])
    full = probability_view(logs,"full")
    assert np.allclose(alr_features(full,[0,2,5]),logs[:,[2,5]]-logs[:,[0]])


@pytest.mark.parametrize("regime,expected",[("independent",0.),("correlated",1-(-.9*math.log2(.9)-.1*math.log2(.1))),("essential",1.)])
def test_solver_recovers_known_zero_error_limits(regime,expected):
    result = solve(binary_problem(regime),1.,0.)
    assert result["leakage_bits"] == pytest.approx(expected,abs=1e-5)
    assert result["numerical_gap_bits"] < 1e-5
    assert result["task_error"] < 1e-7


def test_solver_infeasible_and_no_disclosure():
    p = binary_problem("essential")
    assert solve(p,0.,0.)["status"] == "infeasible"
    assert solve(p,0.,.5)["leakage_bits"] == pytest.approx(0.,abs=1e-7)


def test_conditional_leakage_gradient_and_data_processing():
    p = binary_problem("correlated")
    c = np.random.default_rng(4).dirichlet(np.ones(3),len(p.probabilities))
    _,g = leakage(p,c,True)
    from scipy.optimize._numdiff import approx_derivative
    numerical = approx_derivative(lambda x:leakage(p,x.reshape(c.shape)),c.ravel()).ravel()
    assert np.allclose(g,numerical,atol=1e-5)
    o = np.array([[.8,.2],[.4,.6],[.1,.9]])
    assert output_leakage(p,c,o) <= leakage(p,c)+1e-9
    p.public = p.secrets.copy()
    assert leakage(p,c) == pytest.approx(0.,abs=1e-9)


def test_cluster_bootstrap_and_reward_ablation():
    result = cluster_interval([0,0,1,1],["a","a","b","b"],samples=200)
    assert result["clusters"] == 2 and result["mean"] == .5
    assert result == cluster_interval([0,0,1,1],["a","a","b","b"],samples=200)
    assert reward_components(1,.9,10,10,privacy_weight=0)["reward"] == 1
    assert reward_components(1,.9,20,10)["reward"] < 0


def test_configuration_rejects_fake_probability_pilot():
    with pytest.raises(ValueError,match="impersonate"):
        ExperimentConfig(probability_access=("text","full"))


def test_conditional_secret_error_bound_matches_binary_entropy():
    for error in (.1,.25):
        result = solve(binary_problem("essential"),1.,error)
        expected = 1 + error*math.log2(error) + (1-error)*math.log2(1-error)
        assert result["leakage_bits"] == pytest.approx(expected,abs=1e-5)
        assert result["numerical_gap_bits"] < 1e-5


def test_secret_recovery_is_literal_but_ordinary_answers_allow_formatting():
    from leaklab.metrics import exact,answer_accuracy,scores
    assert exact("KEY-ABC","key abc") == 0
    assert exact("KEY-ABC","KEY-ABC") == 1
    assert exact(None,"") == 0
    examples = generate_split("test",6)
    ordinary = next(e for e in examples if e.regime == "independent")
    assert answer_accuracy(ordinary,ordinary.answer.upper()+".") == 1
    essential = next(e for e in examples if e.regime == "essential")
    assert answer_accuracy(essential,essential.answer.lower()) == 0
    guesses = {k:v.lower() for k,v in essential.secrets.items()}
    assert scores(essential,"",essential.answer,guesses,{})["attribute_recovery"]["private_code"] == 0
