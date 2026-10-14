from __future__ import annotations

import sys
from dataclasses import dataclass, field

CHECK = (sys.executable, "-m", "pytest", "-q", "--tb=short")


@dataclass(frozen=True, slots=True)
class EvalTask:
    name: str
    prompt: str
    files: dict[str, str]
    solution: dict[str, str]
    check: tuple[str, ...] = field(default=CHECK)


TASKS: tuple[EvalTask, ...] = (
    EvalTask(
        name="fix_off_by_one",
        prompt="The tests in test_window.py fail. Fix the bug in window.py.",
        files={
            "window.py": "def last_n(items, n):\n    return items[-n - 1:]\n",
            "test_window.py": (
                "from window import last_n\n\n\n"
                "def test_last_n():\n"
                "    assert last_n([1, 2, 3, 4, 5], 2) == [4, 5]\n\n\n"
                "def test_last_n_all():\n"
                "    assert last_n([1, 2], 5) == [1, 2]\n"
            ),
        },
        solution={"window.py": "def last_n(items, n):\n    return items[-n:] if n > 0 else []\n"},
    ),
    EvalTask(
        name="implement_slugify",
        prompt="Implement slugify in slug.py so that test_slug.py passes.",
        files={
            "slug.py": "def slugify(text):\n    raise NotImplementedError\n",
            "test_slug.py": (
                "from slug import slugify\n\n\n"
                "def test_basic():\n"
                "    assert slugify('Hello World') == 'hello-world'\n\n\n"
                "def test_punctuation_and_spaces():\n"
                "    assert slugify('  Hello,   World!  ') == 'hello-world'\n\n\n"
                "def test_empty():\n"
                "    assert slugify('!!!') == ''\n"
            ),
        },
        solution={
            "slug.py": (
                "import re\n\n\n"
                "def slugify(text):\n"
                "    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n"
            )
        },
    ),
    EvalTask(
        name="fix_mutable_default",
        prompt="test_tags.py fails when add_tag is called twice. Find and fix the cause.",
        files={
            "tags.py": ("def add_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags\n"),
            "test_tags.py": (
                "from tags import add_tag\n\n\n"
                "def test_independent_calls():\n"
                "    assert add_tag('a') == ['a']\n"
                "    assert add_tag('b') == ['b']\n\n\n"
                "def test_explicit_list():\n"
                "    assert add_tag('c', ['x']) == ['x', 'c']\n"
            ),
        },
        solution={
            "tags.py": (
                "def add_tag(tag, tags=None):\n"
                "    tags = [] if tags is None else tags\n"
                "    tags.append(tag)\n"
                "    return tags\n"
            )
        },
    ),
    EvalTask(
        name="rename_across_files",
        prompt=(
            "Rename calc_total to total_price in cart.py and report.py, keeping behavior the "
            "same. test_cart.py already uses the new name."
        ),
        files={
            "cart.py": "def calc_total(prices):\n    return sum(prices)\n",
            "report.py": (
                "from cart import calc_total\n\n\n"
                "def summary(prices):\n    return f'total={calc_total(prices)}'\n"
            ),
            "test_cart.py": (
                "from cart import total_price\nfrom report import summary\n\n\n"
                "def test_total():\n    assert total_price([1, 2, 3]) == 6\n\n\n"
                "def test_summary():\n    assert summary([1, 2]) == 'total=3'\n"
            ),
        },
        solution={
            "cart.py": "def total_price(prices):\n    return sum(prices)\n",
            "report.py": (
                "from cart import total_price\n\n\n"
                "def summary(prices):\n    return f'total={total_price(prices)}'\n"
            ),
        },
    ),
)
