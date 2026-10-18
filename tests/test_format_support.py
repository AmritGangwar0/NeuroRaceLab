import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pytest
from format_support import export_nrlmodel, export_report, import_model_file, validate_nrlmodel


def test_nrlmodel_roundtrip(tmp_path: Path):
    checkpoint = tmp_path / "residual_rnn_seed123_checkpoint.npz"
    np.savez_compressed(checkpoint, parameters=np.zeros(898, dtype=np.float64))
    history = tmp_path / "residual_rnn_seed123_history.json"
    history.write_text(json.dumps({"architecture": "residual_rnn", "hidden_size": 16, "seed": 123}))
    bundle = tmp_path / "model.nrlmodel"
    export_nrlmodel(checkpoint, bundle, history)
    manifest = validate_nrlmodel(bundle)
    assert manifest["architecture"] == "residual_rnn"
    assert manifest["parameter_count"] == 898
    library = tmp_path / "library"
    copied = import_model_file(bundle, library)
    assert copied.exists()
    with zipfile.ZipFile(copied) as zf:
        assert {"manifest.json", "parameters.npz", "metadata.json"} <= set(zf.namelist())


def test_rejects_wrong_nrlmodel(tmp_path: Path):
    bad = tmp_path / "bad.nrlmodel"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("hello.txt", "nope")
    with pytest.raises(ValueError):
        validate_nrlmodel(bad)


def test_csv_to_pdf_docx_pptx(tmp_path: Path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("reportlab")
    pytest.importorskip("docx")
    pytest.importorskip("pptx")
    source = tmp_path / "history.csv"
    source.write_text("generation,best_fitness\n1,0.2\n2,0.7\n3,0.9\n")
    for fmt in ("pdf", "docx", "pptx"):
        output = tmp_path / f"report.{fmt}"
        export_report(source, output, fmt)
        assert output.is_file() and output.stat().st_size > 100
    assert (tmp_path / "report_plot.png").is_file()
