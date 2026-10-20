from eurus.dotenv import is_allowed, load_dotenv, parse_dotenv


def test_parse_dotenv_handles_quotes_export_and_comments():
    text = "\n".join(
        [
            "# comment",
            "GROQ_API_KEY=gsk_one",
            'export NVIDIA_API_KEY="nvapi-two"',
            "OPENAI_API_KEY='sk-three'  ",
            "XAI_API_KEY=plain # trailing note",
            "broken line",
            "=novalue",
        ]
    )
    values = parse_dotenv(text)
    assert values == {
        "GROQ_API_KEY": "gsk_one",
        "NVIDIA_API_KEY": "nvapi-two",
        "OPENAI_API_KEY": "sk-three",
        "XAI_API_KEY": "plain",
    }


def test_only_provider_keys_are_loaded(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "GROQ_API_KEY=gsk_x\nEURUS_PERMISSION_MODE=yolo\nNVIDIA_BASE_URL=http://h/v1\nPATH=/evil\n"
    )
    environ = {"PATH": "/bin"}
    assert load_dotenv([path], environ) == [path]
    assert environ == {
        "PATH": "/bin",
        "GROQ_API_KEY": "gsk_x",
        "NVIDIA_BASE_URL": "http://h/v1",
    }


def test_existing_values_win_but_empty_ones_are_filled(tmp_path):
    path = tmp_path / ".env"
    path.write_text("GROQ_API_KEY=from_file\nOPENAI_API_KEY=from_file\n")
    environ = {"GROQ_API_KEY": "from_env", "OPENAI_API_KEY": "  "}
    load_dotenv([path], environ)
    assert environ["GROQ_API_KEY"] == "from_env"
    assert environ["OPENAI_API_KEY"] == "from_file"


def test_missing_file_is_ignored(tmp_path):
    assert load_dotenv([tmp_path / "none.env"], {}) == []
    assert is_allowed("MYHOST_API_KEY")
    assert not is_allowed("EURUS_MODEL")
