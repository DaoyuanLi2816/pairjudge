"""Actual optimizer/weights/loader tests, without network access."""

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from pairjudge import PackerConfig, PairwiseJudge, from_pairs
from pairjudge.artifacts import read_manifest
from pairjudge.training import JudgeTrainConfig, summed_loss, train


@pytest.mark.parametrize(
    "mode,lora", [("hard", False), ("hard", True), ("soft", False), ("soft", True)]
)
def test_real_roundtrip(tiny_base, tmp_path, mode, lora):
    frame = from_pairs(
        [f"q {i}" for i in range(12)],
        [f"good a {i}" for i in range(12)],
        [f"bad b {i}" for i in range(12)],
        ["a", "b", "tie"] * 4,
    )
    if mode == "soft":
        frame[["winner_model_a", "winner_model_b", "winner_tie"]] = np.tile(
            [0.6, 0.3, 0.1], (12, 1)
        )
    path = tmp_path / "train.jsonl"
    frame.to_json(path, orient="records", lines=True)
    output = tmp_path / "output"
    cfg = JudgeTrainConfig(
        train_path=str(path),
        model_name=str(tiny_base),
        output_dir=str(output),
        label_mode=mode,
        use_lora=lora,
        lora_target_modules=["c_attn"],
        lora_r=2,
        lora_alpha=4,
        max_length=100,
        packer={
            "ratios": [0.3, 0.35, 0.35],
            "min_tail_budget": 0,
            "final_instruction": "judge q",
            "ellipsis": "cut",
        },
        label_order=["tie", "a_wins", "b_wins"],
        device="cpu",
        max_steps=2,
        lr=0.005,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=3,
        eval_holdout=0.25,
    )
    result = train(cfg)
    assert result["optimizer_steps"] == 2 and result["head_changed"]
    judge = PairwiseJudge.from_pretrained(
        output / "merged", device="cpu", local_files_only=True
    )
    predictions = judge.predict_proba(frame, batch_size=5)
    assert predictions.shape == (12, 3) and np.isfinite(predictions).all()
    assert judge.label_order == ("tie", "a_wins", "b_wins")
    assert judge.packer.config.ratios == (0.3, 0.35, 0.35)
    assert judge.packer.config.final_instruction == "judge q"
    second = tmp_path / "saved"
    judge.save_pretrained(second)
    restored = PairwiseJudge.from_pretrained(second, device="cpu")
    np.testing.assert_allclose(
        predictions, restored.predict_proba(frame, batch_size=5), atol=1e-7
    )
    if lora:
        adapter = PairwiseJudge.from_pretrained(
            output / "adapter", device="cpu", local_files_only=True
        )
        np.testing.assert_allclose(
            predictions, adapter.predict_proba(frame, batch_size=5), atol=2e-6
        )
    code = "from pairjudge import PairwiseJudge,from_pairs; import json,sys; j=PairwiseJudge.from_pretrained(sys.argv[1],device='cpu',local_files_only=True); print(json.dumps(j.predict_proba(from_pairs(['q 0'],['good a 0'],['bad b 0'])).tolist()))"
    fresh = subprocess.run(
        [sys.executable, "-c", code, str(output / ("adapter" if lora else "merged"))],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    np.testing.assert_allclose(json.loads(fresh.stdout), predictions[:1], atol=2e-6)
    with pytest.raises(FileExistsError):
        train(cfg)
    config = json.loads((second / "config.json").read_text())
    config["id2label"]["0"] = "a_wins"
    (second / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_manifest(second)


def test_backbone_rejected(tiny_base):
    with pytest.raises(ValueError, match="missing pairjudge.json"):
        PairwiseJudge.from_pretrained(tiny_base, device="cpu")
    with pytest.raises(ValueError, match="incomplete trained weights"):
        PairwiseJudge.from_pretrained(
            tiny_base,
            device="cpu",
            allow_legacy=True,
            packer_config=PackerConfig(),
            label_order=["a_wins", "b_wins", "tie"],
        )


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_accumulation_matches_full_batch_gradients(mode):
    torch.manual_seed(8)
    full = torch.nn.Linear(4, 3)
    accumulated = torch.nn.Linear(4, 3)
    accumulated.load_state_dict(full.state_dict())
    x = torch.randn(7, 4)
    labels = (
        torch.tensor([0, 1, 2, 0, 1, 2, 0])
        if mode == "hard"
        else torch.softmax(torch.randn(7, 3), -1)
    )
    (summed_loss(full(x), labels, mode) / 7).backward()
    for start in range(0, 7, 3):
        (
            summed_loss(
                accumulated(x[start : start + 3]), labels[start : start + 3], mode
            )
            / 7
        ).backward()
    for left, right in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-6, atol=1e-7)


def test_invalid_input_and_labels_rejected(tokenizer):
    from pairjudge import PairPacker

    packer = PairPacker(tokenizer)
    with pytest.raises(ValueError, match="scalar string"):
        packer.pack("q", "a", "b")
    with pytest.raises(ValueError, match="null"):
        packer.pack([None], ["a"], ["b"])
    with pytest.raises(ValueError, match="positive"):
        PackerConfig(ratios=[1, 0, 0])
    from pairjudge.validation import decisions

    assert decisions([[0.4, 0.4, 0.2], [0.2, 0.2, 0.6]]) == ["ambiguous", "tie"]
