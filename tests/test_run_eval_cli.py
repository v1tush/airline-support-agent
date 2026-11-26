import io
import sys

from scripts.run_eval import _configure_stdout


def test_console_output_escapes_characters_missing_from_cp1251(monkeypatch):
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1251", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)

    _configure_stdout()
    print("MOW → KZN")
    stream.flush()

    output = buffer.getvalue().decode("cp1251")
    assert output.replace("\r\n", "\n") == "MOW \\u2192 KZN\n"
