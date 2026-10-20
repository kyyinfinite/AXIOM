import pytest

from eurus.providers.policy import is_local_url


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:11434/v1",
        "http://[::1]:8080/v1",
        "http://192.168.1.20:8000/v1",
        "http://10.0.0.5/v1",
        "http://100.101.102.103:11434/v1",
        "http://box.local:8000/v1",
    ],
)
def test_local_urls(url):
    assert is_local_url(url)


@pytest.mark.parametrize(
    "url",
    ["https://api.openai.com/v1", "https://8.8.8.8/v1", "https://example.com", "not a url"],
)
def test_remote_urls(url):
    assert not is_local_url(url)
