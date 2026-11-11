import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.compute_iaa import check_files_distinct, compute_fleiss_kappa, compute_iaa


def test_refuse_identical_files(tmp_path: Path):
    f1 = tmp_path / "annotator_1.csv"
    f2 = tmp_path / "annotator_2.csv"
    content = "claim,label\nclaim1,supported\nclaim2,hallucinated\n"
    f1.write_text(content, encoding="utf-8")
    f2.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="byte-identical"):
        check_files_distinct([f1, f2])


def test_refuse_empty_file(tmp_path: Path):
    f1 = tmp_path / "annotator_1.csv"
    f2 = tmp_path / "annotator_2.csv"
    f1.write_text("claim,label\nclaim1,supported\n", encoding="utf-8")
    f2.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        check_files_distinct([f1, f2])


def test_compute_iaa_perfect_agreement(tmp_path: Path):
    f1 = tmp_path / "a1.csv"
    f2 = tmp_path / "a2.csv"
    f3 = tmp_path / "a3.csv"
    # Same labels, but distinct file content (different comment or claim text) to ensure genuine independent files
    f1.write_text("# rater 1\nlabel\nsupported\nhallucinated\nsupported\n", encoding="utf-8")
    f2.write_text("# rater 2\nlabel\nsupported\nhallucinated\nsupported\n", encoding="utf-8")
    f3.write_text("# rater 3\nlabel\nsupported\nhallucinated\nsupported\n", encoding="utf-8")

    res = compute_iaa([f1, f2, f3])
    assert res["n_annotators"] == 3
    assert res["n_samples"] == 3
    assert np.isclose(res["fleiss_kappa"], 1.0)
    assert np.isclose(res["mean_cohen_kappa"], 1.0)


def test_compute_iaa_partial_agreement(tmp_path: Path):
    f1 = tmp_path / "a1.csv"
    f2 = tmp_path / "a2.csv"
    f1.write_text("label\nsupported\nsupported\nhallucinated\nhallucinated\n", encoding="utf-8")
    f2.write_text("label\nsupported\nhallucinated\nhallucinated\nsupported\n", encoding="utf-8")

    res = compute_iaa([f1, f2])
    assert res["n_annotators"] == 2
    assert res["n_samples"] == 4
    assert res["pairwise_cohen_kappa"][0]["percent_agreement"] == 0.5
