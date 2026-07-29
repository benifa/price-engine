"""Tests for Ollama export staging (no merge / no network)."""

from pathlib import Path

from priceengine.training.ollama_export import prepare_ollama_export


def test_prepare_ollama_export_writes_modelfile(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    out = tmp_path / "ollama"
    result = prepare_ollama_export(
        adapter, out_dir=out, gguf_name="custom.gguf"
    )
    assert Path(result["modelfile"]).is_file()
    text = Path(result["modelfile"]).read_text()
    assert "FROM ./custom.gguf" in text
    recipe = Path(result["recipe"]).read_text()
    assert "merge_and_unload" in recipe
    assert "ollama create" in recipe
