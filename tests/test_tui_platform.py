import zipfile

import pytest

from eurus.tui import fonts, themes
from eurus.tui.commands import COMMANDS
from eurus.tui.commands import resolve as resolve_command
from eurus.tui.theme import ASCII_GLYPHS, NERD_GLYPHS, UNICODE_GLYPHS, Theme

TTF = b"\x00\x01\x00\x00" + b"x" * 64


def write_theme(home, name, body):
    folder = home / "themes"
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.toml").write_text(body, encoding="utf-8")


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)


def test_command_names_and_themes_do_not_collide():
    assert [item.name for item in resolve_command("/theme")] == ["/theme"]
    assert [item.name for item in resolve_command("/font")] == ["/font"]
    assert len({item.name for item in COMMANDS}) == len(COMMANDS)
    with pytest.raises(ValueError):
        themes.resolve("/theme")


def test_all_builtin_themes_resolve_with_valid_code_styles():
    for name in themes.BUILTIN:
        theme = themes.resolve(name)
        assert theme.code_theme in themes.code_styles(), name
        assert name in themes.DESCRIPTIONS


def test_extends_glyphs_code_and_font(tmp_path):
    write_theme(
        tmp_path,
        "ocean",
        '[meta]\ndescription = "Deep"\nextends = "nord"\n'
        '[colors]\ncopper = "#112233"\n'
        '[style]\ncode = "dracula"\nglyphs = "nerd"\n[style.glyph]\nprompt = "$"\nbad = "x"\n'
        '[font]\nname = "hack-nerd"\n',
    )
    theme = themes.resolve("ocean", home=tmp_path)
    assert theme.copper == "#112233"
    assert theme.background == themes.resolve("nord").background
    assert theme.code_theme == "dracula"
    assert theme.glyphs.prompt == "$" and theme.glyphs.dot == NERD_GLYPHS.dot
    assert themes.font_for("ocean", tmp_path) == "hack-nerd"
    assert themes.describe_theme("ocean", tmp_path) == "Deep"
    ascii_theme = themes.resolve("ocean", unicode=False, home=tmp_path)
    assert ascii_theme.glyphs.dot == ASCII_GLYPHS.dot and not ascii_theme.unicode


def test_inheritance_loop_and_unknown_parent_are_rejected(tmp_path):
    write_theme(tmp_path, "a", '[meta]\nextends = "b"\n')
    write_theme(tmp_path, "b", '[meta]\nextends = "a"\n')
    write_theme(tmp_path, "c", '[meta]\nextends = "ghost"\n')
    for name in ("a", "c"):
        with pytest.raises(ValueError):
            themes.resolve(name, home=tmp_path)


def test_invalid_style_values_are_ignored(tmp_path):
    write_theme(
        tmp_path,
        "odd",
        '[style]\ncode = "no-such-style"\nglyphs = "weird"\n'
        '[style.glyph]\nprompt = "\\u001b[31m"\ndot = "waytoolongglyphvalue"\n',
    )
    theme = themes.resolve("odd", home=tmp_path)
    assert theme.code_theme == Theme().code_theme
    assert theme.glyphs == UNICODE_GLYPHS


def test_glyph_choice_precedence_and_ascii_fallback(tmp_path):
    assert themes.glyph_choice({}, tmp_path) == ""
    themes.save_glyphs("nerd", tmp_path)
    assert themes.glyph_choice({}, tmp_path) == "nerd"
    assert themes.glyph_choice({"EURUS_GLYPHS": "ascii"}, tmp_path) == "ascii"
    assert themes.load_active(True, {}, tmp_path).glyphs == NERD_GLYPHS
    assert themes.load_active(False, {}, tmp_path).glyphs == ASCII_GLYPHS
    themes.save_glyphs("", tmp_path)
    assert themes.glyph_choice({}, tmp_path) == ""


def test_scaffold_creates_loadable_theme_and_refuses_bad_names(tmp_path):
    path = themes.scaffold("Mine", "dracula", tmp_path)
    assert path.name == "mine.toml"
    theme = themes.resolve("mine", home=tmp_path)
    assert theme.background == themes.resolve("dracula").background
    assert theme.code_theme == "dracula"
    for bad in ("mine", "eurus", "preview", "../x", "a b", ""):
        with pytest.raises(ValueError):
            themes.scaffold(bad, "eurus", tmp_path)
    assert "preview" not in themes.available(tmp_path)


def test_termux_colors_cover_core_keys_and_backup(tmp_path):
    theme = themes.resolve("nord")
    text = themes.termux_colors(theme)
    assert f"background={theme.background}" in text and "color15=" in text
    target = tmp_path / ".termux" / "colors.properties"
    target.parent.mkdir()
    target.write_text("old\n", encoding="utf-8")
    themes.write_termux_colors(theme, target)
    themes.write_termux_colors(theme, target)
    backup = target.with_name("colors.properties.eurus-backup")
    assert backup.read_text(encoding="utf-8") == "old\n"
    assert target.read_text(encoding="utf-8") == text


def test_font_target_detection(tmp_path):
    termux = fonts.detect_target({"TERMUX_VERSION": "1"}, "linux", tmp_path)
    assert termux.kind == "termux" and termux.path == tmp_path / ".termux" / "font.ttf"
    linux = fonts.detect_target({}, "linux", tmp_path, "A.ttf")
    assert linux.kind == "linux" and linux.path.name == "A.ttf"
    assert fonts.detect_target({}, "darwin", tmp_path, "A.ttf").kind == "macos"
    assert fonts.detect_target({}, "win32", tmp_path).path is None


def test_extract_font_picks_mono_regular_and_rejects_junk(tmp_path):
    archive = tmp_path / "f.zip"
    make_zip(
        archive,
        {
            "X-Bold.ttf": TTF,
            "XNerdFont-Regular.ttf": TTF,
            "XNerdFontMono-Regular.ttf": TTF + b"1",
            "../evil/XNerdFontMono-Regular.ttf.txt": b"no",
        },
    )
    name, data = fonts.extract_font(archive, "*NerdFontMono-Regular.ttf")
    assert name == "XNerdFontMono-Regular.ttf" and data.endswith(b"1")
    with pytest.raises(fonts.FontError):
        fonts.extract_font(archive, "*.woff")
    bad = tmp_path / "bad.zip"
    make_zip(bad, {"XNerdFontMono-Regular.ttf": b"not a font"})
    with pytest.raises(fonts.FontError):
        fonts.extract_font(bad, "*NerdFontMono-Regular.ttf")
    plain = tmp_path / "plain.zip"
    plain.write_bytes(b"nope")
    with pytest.raises(fonts.FontError):
        fonts.extract_font(plain, "*.ttf")


def test_install_termux_backs_up_reloads_and_resets(tmp_path):
    target = fonts.detect_target({"TERMUX_VERSION": "1"}, "linux", tmp_path)
    target.path.parent.mkdir()
    target.path.write_bytes(b"\x00\x01\x00\x00original")
    calls = []

    def runner(command):
        calls.append(command)
        return True

    result = fonts.install("F.ttf", TTF, target, runner)
    assert result.reloaded and calls == [["termux-reload-settings"]]
    assert target.path.read_bytes() == TTF
    assert not target.path.with_name("font.ttf.part").exists()
    fonts.install("G.ttf", TTF + b"2", target, runner)
    backup = target.path.with_name("font.ttf.eurus-backup")
    assert backup.read_bytes().endswith(b"original")
    assert "restored" in fonts.reset(target, runner)
    assert target.path.read_bytes().endswith(b"original")


def test_install_rejects_non_font_and_manual_target(tmp_path):
    target = fonts.detect_target({}, "linux", tmp_path, "A.ttf")
    with pytest.raises(fonts.FontError):
        fonts.install("A.ttf", b"zzzz", target, lambda command: True)
    with pytest.raises(fonts.FontError):
        fonts.install("A.ttf", TTF, fonts.FontTarget("manual", None, "by hand"))
    result = fonts.install("A.ttf", TTF, target, lambda command: True)
    assert result.path.read_bytes() == TTF
    assert "A.ttf" in fonts.status(target)


def test_fetch_uses_injected_downloader_and_catalog_lookup(tmp_path):
    spec = fonts.lookup("Hack-Nerd")

    def fake(url, destination):
        assert url.startswith("https://")
        make_zip(destination, {"HackNerdFontMono-Regular.ttf": TTF})

    assert fonts.fetch(spec, fake)[0] == "HackNerdFontMono-Regular.ttf"
    with pytest.raises(fonts.FontError):
        fonts.lookup("comic-sans")
    with pytest.raises(fonts.FontError):
        fonts.download("http://example.com/f.zip", tmp_path / "x")


def test_read_local_accepts_font_and_zip(tmp_path):
    ttf = tmp_path / "My.ttf"
    ttf.write_bytes(TTF)
    assert fonts.read_local(ttf)[0] == "My.ttf"
    archive = tmp_path / "pack.zip"
    make_zip(archive, {"inner/My.ttf": TTF})
    assert fonts.read_local(archive, "*.ttf")[0] == "My.ttf"
    other = tmp_path / "x.txt"
    other.write_text("hi", encoding="utf-8")
    for bad in (other, tmp_path / "missing.ttf"):
        with pytest.raises(fonts.FontError):
            fonts.read_local(bad)


def test_install_source_from_catalog_and_local_file(tmp_path):
    env = {"HOME": str(tmp_path), "TERMUX_VERSION": "1"}

    def fake(url, destination):
        make_zip(destination, {"JetBrainsMonoNerdFontMono-Regular.ttf": TTF})

    result = fonts.install_source("jetbrains-mono-nerd", env, fake, lambda command: True)
    assert result.path == tmp_path / ".termux" / "font.ttf" and result.reloaded
    local = tmp_path / "Mine.ttf"
    local.write_bytes(TTF + b"3")
    fonts.install_source(str(local), env, fake, lambda command: False)
    assert result.path.read_bytes().endswith(b"3")
    with pytest.raises(fonts.FontError):
        fonts.install_source("nope-font", env, fake, lambda command: True)
