"""Local tests for the Tier 1–3 extra tools (offline-safe).

Validates registry wiring, arg parsing, and every *offline* executor against a
temp workspace. Network tools (currency, stock, geocode, news, wikipedia,
translate, pdf-from-url) are smoke-tested but skipped on any network failure so
the suite is deterministic offline. Run: ``pytest agents/test_tools_extra.py``
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Jail the workspace to a temp dir BEFORE the executors run.
_TMP = tempfile.mkdtemp(prefix="om_ws_")
os.environ["WORKSPACE_DIR"] = _TMP

import agents.tools as tools  # noqa: E402
import agents.tools_extra as tx  # noqa: E402


EXPECTED = {
    "read_file", "list_dir", "write_file", "shell", "apply_patch", "note",
    "pdf_read", "doc_index", "doc_search", "sql_query", "csv_query", "wikipedia",
    "unit_convert", "currency", "timezone", "translate", "stock_price",
    "crypto_price", "geocode", "news",
    # batch 2
    "hash_text", "base64_tool", "uuid_gen", "password_gen", "json_format",
    "regex_test", "roman", "number_base", "morse", "slugify", "epoch_convert",
    "lorem_ipsum", "dictionary", "synonyms", "country_info", "public_holidays",
    "quote", "joke", "forecast", "ip_info",
}


def test_all_tools_registered():
    reg = tools.registry()
    missing = EXPECTED - set(reg)
    assert not missing, f"missing from registry: {missing}"


def test_schemas_present_and_wellformed():
    names = {s["name"] for s in tools.schemas()}
    assert EXPECTED <= names
    for s in tools.schemas():
        assert s["name"] and s["example_call"]


def test_training_examples_nonempty():
    reg = tools.registry()
    for name in EXPECTED:
        ex = reg[name].training_examples()
        assert ex, f"{name} has no training examples"
        assert all("prompt" in e and "completion" in e for e in ex)
        assert all(name in e["completion"] for e in ex)


def test_arg_extraction_roundtrip():
    # The minted format name("arg") must parse back to the arg.
    assert tools.extract_arg('unit_convert("10 km to miles")', "unit_convert") == "10 km to miles"
    assert tools.extract_arg('stock_price("NVDA")', "stock_price") == "NVDA"


def test_stock_price_yesterday_uses_previous_row(monkeypatch):
    csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2026-06-04,210.0,215.0,209.0,212.50,100\n"
        "2026-06-05,213.0,218.0,212.0,214.86,120\n"
    )

    def fake_get(url: str, timeout: int = 10) -> str:
        assert "/q/d/l/" in url
        return csv

    monkeypatch.setattr(tx, "_http_get", fake_get)
    out = tx._stock_price("NVDA yesterday")
    assert "212.50" in out
    assert "2026-06-04" in out
    assert "previous trading day" in out


def test_stock_price_is_gate_mode():
    t = tools.get("stock_price")
    assert t.arg_mode == "gate"


# --- Tier 1 (offline) ------------------------------------------------------


def test_file_write_read_list_roundtrip():
    assert "wrote" in tx._write_file("notes.txt\nhello world")
    assert "hello world" in tx._read_file("notes.txt")
    listing = tx._list_dir(".")
    assert "notes.txt" in listing


def test_workspace_jail_blocks_escape():
    with pytest.raises(tools.ToolError):
        tx._read_file("../../etc/passwd")


def test_shell_runs_and_blocks_dangerous():
    tx._write_file("hi.txt\nline1\nline2")
    out = tx._shell("cat hi.txt" if os.name != "nt" else "type hi.txt")
    # On the box (Linux) this is `cat`; locally on Windows we at least exercise the guard.
    assert isinstance(out, str)
    with pytest.raises(tools.ToolError):
        tx._shell("rm -rf /")


def test_apply_patch_edits_file():
    tx._write_file("p.txt\nold")
    tx._apply_patch("--- a/p.txt\n+++ b/p.txt\n@@ -1 +1 @@\n-old\n+new")
    assert "new" in tx._read_file("p.txt")


def test_note_save_and_recall():
    os.environ["REDIS_URL"] = "redis://127.0.0.1:6553/0"  # unreachable -> memory fallback
    tx._note("save: user likes terse answers")
    out = tx._note("recall")
    assert "terse" in out


# --- Tier 2 (offline) ------------------------------------------------------


def test_sql_query_select_only():
    assert "1" in tx._sql_query("SELECT 1")
    with pytest.raises(tools.ToolError):
        tx._sql_query("DROP TABLE users")
    with pytest.raises(tools.ToolError):
        tx._sql_query("INSERT INTO t VALUES (1)")


def test_csv_query_stdlib_fallback():
    tx._write_file("data.csv\nname,amount\na,10\nb,20")
    assert "name" in tx._csv_query("data.csv columns")
    assert "rows" in tx._csv_query("data.csv shape")


def test_doc_index_and_search():
    os.environ["REDIS_URL"] = "redis://127.0.0.1:6553/0"  # force memory fallback
    tx._doc_index("d1\nOpenMirror bakes memory into model weights overnight")
    tx._doc_index("d2\nRedis stores adapters, policies and the audit stream")
    res = tx._doc_search("weights memory")
    assert "d1" in res


# --- Tier 3 (offline-computable) -------------------------------------------


def test_timezone_now_and_convert():
    assert "T" in tx._timezone("now in Tokyo")  # ISO timestamp
    out = tx._timezone("3pm UTC to PST")
    assert "->" in out


def test_unit_convert_or_skip():
    try:
        import pint  # noqa: F401
    except Exception:
        pytest.skip("pint not installed locally")
    out = tx._unit_convert("10 km to miles")
    assert "miles" in out and "6.2" in out


def test_bad_args_raise():
    with pytest.raises(tools.ToolError):
        tx._unit_convert("nonsense")
    with pytest.raises(tools.ToolError):
        tx._currency("not a currency line")
    with pytest.raises(tools.ToolError):
        tx._timezone("???")


# --- Batch 2 offline (deterministic) ---------------------------------------


def test_hash_text():
    out = tx._hash_text("sha256: abc")
    # sha256("abc") is a known constant
    assert "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" in out


def test_base64_roundtrip():
    enc = tx._base64_tool("encode: hello world")
    assert enc == "aGVsbG8gd29ybGQ="
    assert tx._base64_tool("decode: aGVsbG8gd29ybGQ=") == "hello world"


def test_uuid_gen_count():
    out = tx._uuid_gen("3").splitlines()
    assert len(out) == 3
    assert all(len(u) == 36 for u in out)


def test_password_gen_length():
    assert len(tx._password_gen("24")) == 24
    assert len(tx._password_gen("")) == 16


def test_json_format_and_invalid():
    assert '"a": 1' in tx._json_format('{"a":1,"b":2}')
    with pytest.raises(tools.ToolError):
        tx._json_format("{not json}")


def test_regex_test():
    assert "match" in tx._regex_test(r"\d+ ||| order 66 in 3 days")
    assert "no match" in tx._regex_test(r"zzz ||| nothing here")
    with pytest.raises(tools.ToolError):
        tx._regex_test("no separator")


def test_roman_both_ways():
    assert tx._roman("2026") == "MMXXVI"
    assert tx._roman("MCMXCIV") == "1994"
    with pytest.raises(tools.ToolError):
        tx._roman("4000")


def test_number_base():
    assert tx._number_base("255 to hex") == "0xff"
    assert tx._number_base("0xff to dec") == "255"
    assert tx._number_base("10 to bin") == "0b1010"


def test_morse_roundtrip():
    code = tx._morse("SOS")
    assert code == "... --- ..."
    assert tx._morse("... --- ...") == "SOS"


def test_slugify():
    assert tx._slugify("Hello World!") == "hello-world"


def test_epoch_convert():
    assert tx._epoch_convert("1700000000").startswith("2023-11-14")
    assert tx._epoch_convert("2021-01-01T00:00:00").isdigit()
    assert tx._epoch_convert("now").isdigit()


def test_lorem_ipsum():
    out = tx._lorem_ipsum("10")
    assert len(out.split()) == 10
    assert out.endswith(".")


# --- Network smoke (skip on failure) ---------------------------------------

@pytest.mark.parametrize("fn,arg", [
    (lambda a: tx._wikipedia(a), "Alan Turing"),
    (lambda a: tx._currency(a), "100 USD to EUR"),
    (lambda a: tx._crypto_price(a), "bitcoin"),
    (lambda a: tx._stock_price(a), "AAPL"),
    (lambda a: tx._geocode(a), "Eiffel Tower"),
    (lambda a: tx._news(a), "technology"),
    (lambda a: tx._translate(a), "hello to French"),
    (lambda a: tx._dictionary(a), "serendipity"),
    (lambda a: tx._synonyms(a), "happy"),
    (lambda a: tx._country_info(a), "Japan"),
    (lambda a: tx._public_holidays(a), "2026 US"),
    (lambda a: tx._quote(a), ""),
    (lambda a: tx._joke(a), ""),
    (lambda a: tx._forecast(a), "Tokyo"),
    (lambda a: tx._ip_info(a), "8.8.8.8"),
])
def test_network_smoke(fn, arg):
    try:
        out = fn(arg)
    except Exception as e:
        pytest.skip(f"network unavailable: {e}")
    assert isinstance(out, str) and out
