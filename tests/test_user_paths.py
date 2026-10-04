"""Exercise installed-process CLI and loopback HTTP using actual trained weights."""

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from pairjudge import PairwiseJudge, from_pairs
from pairjudge.cli import input_record
from pairjudge.demo import make_server
from pairjudge.pseudo_label import pseudo_label
from pairjudge.training import JudgeTrainConfig, train


@pytest.fixture
def usable_bundle(tiny_base, tmp_path):
    frame = from_pairs(
        [f"q {i}" for i in range(8)],
        [f"a {i}" for i in range(8)],
        [f"b {i}" for i in range(8)],
        ["a", "b", "tie", "a"] * 2,
    )
    path = tmp_path / "human.jsonl"
    frame.to_json(path, lines=True, orient="records")
    train(
        JudgeTrainConfig(
            train_path=str(path),
            model_name=str(tiny_base),
            output_dir=str(tmp_path / "human"),
            max_length=100,
            use_lora=False,
            device="cpu",
            max_steps=1,
            warmup_steps=0,
        )
    )
    return tmp_path / "human" / "merged"


def test_cli_and_local_demo(usable_bundle, tmp_path):
    records = [
        {"id": identifier, "prompt": text, "response_a": "a", "response_b": "b"}
        for identifier, text in [("z", "q " * 30), ("z", "q"), ("x", "Unicode 中文\nq")]
    ]
    input_path = tmp_path / "pairs.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(row) for row in records), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pairjudge",
            "batch",
            "--model",
            str(usable_bundle),
            "--input",
            str(input_path),
            "--device",
            "cpu",
            "--offline",
            "--swap",
            "--batch-size",
            "2",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    output = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["id"] for row in output] == ["z", "z", "x"]
    assert output[0]["model"]["mode"] == "swap_average"
    judge = PairwiseJudge.from_pretrained(usable_bundle, device="cpu")
    frame = pd.DataFrame([input_record(row) for row in records], index=[7, 7, 4])
    expected = judge.predict_proba(frame, swap_debias=True, batch_size=2)
    np.testing.assert_allclose(
        [
            [row["probabilities"][name] for name in ("a_wins", "b_wins", "tie")]
            for row in output
        ],
        expected,
        atol=1e-7,
    )
    server = make_server(judge, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        html = urllib.request.urlopen(url).read().decode()
        assert "textContent" in html and "innerHTML" not in html
        payload = json.dumps(
            {
                "prompt": "<script>alert(1)</script>",
                "response_a": "a",
                "response_b": "b",
            }
        ).encode()
        request = urllib.request.Request(
            url + "/compare", data=payload, headers={"Content-Type": "application/json"}
        )
        response = json.load(urllib.request.urlopen(request))
        assert sum(response["average"]) == pytest.approx(1.0)
        bad = urllib.request.Request(
            url + "/compare",
            data=payload,
            headers={"Origin": "http://malicious.example"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad)
        assert error.value.code == 403
        bad = urllib.request.Request(url + "/compare", data=b"x" * 180001)
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad)
        assert error.value.code == 413
    finally:
        server.shutdown()
        server.server_close()


def test_real_human_pseudo_soft_chain(usable_bundle, tiny_base, tmp_path):
    judge = PairwiseJudge.from_pretrained(usable_bundle, device="cpu")
    pool = from_pairs(
        [f"pool q {i}" for i in range(8)],
        [f"good {i}" for i in range(8)],
        [f"bad {i}" for i in range(8)],
    )
    soft = pseudo_label(judge, pool, swap_debias=True)
    path = tmp_path / "pseudo.jsonl"
    soft.to_json(path, orient="records", lines=True)
    result = train(
        JudgeTrainConfig(
            train_path=str(path),
            model_name=str(tiny_base),
            output_dir=str(tmp_path / "student"),
            label_mode="soft",
            use_lora=True,
            lora_target_modules=["c_attn"],
            max_length=100,
            device="cpu",
            max_steps=1,
            warmup_steps=0,
        )
    )
    assert result["head_changed"]
    student = PairwiseJudge.from_pretrained(
        tmp_path / "student" / "adapter", device="cpu"
    )
    assert np.isfinite(student.predict_proba(pool)).all()
    assert soft.attrs["pairjudge_teacher"]["artifact_sha256"]


def test_bad_data_fails_before_loading(tmp_path):
    frame = from_pairs(["q"], ["a"], ["b"], ["a"])
    frame["winner_tie"] = np.nan
    path = tmp_path / "bad.jsonl"
    frame.to_json(path, lines=True, orient="records")
    with pytest.raises(ValueError, match="finite"):
        train(
            JudgeTrainConfig(
                train_path=str(path),
                model_name="must-not-download",
                output_dir=str(tmp_path / "out"),
            )
        )
    assert not (tmp_path / "out").exists()


def test_precision_and_incompatible_bundle_requests(usable_bundle):
    import torch

    from pairjudge import PackerConfig
    from pairjudge.judge import resolve_precision

    assert resolve_precision("cpu", "auto")[1] == torch.float32
    with pytest.raises(ValueError, match="CPU"):
        resolve_precision("cpu", "float16")
    with pytest.raises(ValueError, match="unqualified"):
        PairwiseJudge.from_pretrained(usable_bundle, device="cpu", load_in_8bit=True)
    with pytest.raises(ValueError, match="override"):
        PairwiseJudge.from_pretrained(
            usable_bundle, device="cpu", packer_config=PackerConfig(max_length=101)
        )
    with pytest.raises(ValueError, match="label_order"):
        PairwiseJudge.from_pretrained(
            usable_bundle, device="cpu", label_order=["tie", "a_wins", "b_wins"]
        )
