from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from eurus.termux import is_termux

MAX_DOWNLOAD = 200 * 1024 * 1024
MAX_FONT = 40 * 1024 * 1024
MAGIC = (b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf")
NERD_BASE = "https://github.com/ryanoasis/nerd-fonts/releases/latest/download/"
BACKUP_SUFFIX = ".eurus-backup"
Runner = Callable[[list[str]], bool]


@dataclass(frozen=True, slots=True)
class FontSpec:
    key: str
    family: str
    archive: str
    pattern: str
    note: str


CATALOG: dict[str, FontSpec] = {
    "jetbrains-mono-nerd": FontSpec(
        "jetbrains-mono-nerd",
        "JetBrainsMono Nerd Font Mono",
        NERD_BASE + "JetBrainsMono.zip",
        "*NerdFontMono-Regular.ttf",
        "Clean, tall, great for code",
    ),
    "fira-code-nerd": FontSpec(
        "fira-code-nerd",
        "FiraCode Nerd Font Mono",
        NERD_BASE + "FiraCode.zip",
        "*NerdFontMono-Regular.ttf",
        "Popular, with programming ligatures",
    ),
    "hack-nerd": FontSpec(
        "hack-nerd",
        "Hack Nerd Font Mono",
        NERD_BASE + "Hack.zip",
        "*NerdFontMono-Regular.ttf",
        "Sturdy and compact",
    ),
    "cascadia-code-nerd": FontSpec(
        "cascadia-code-nerd",
        "CaskaydiaCove Nerd Font Mono",
        NERD_BASE + "CascadiaCode.zip",
        "*NerdFontMono-Regular.ttf",
        "Microsoft Cascadia, rounded",
    ),
}


class FontError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class FontTarget:
    kind: str
    path: Path | None
    note: str


@dataclass(frozen=True, slots=True)
class InstallResult:
    path: Path
    reloaded: bool
    message: str


def user_home(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    custom = env.get("HOME", "").strip()
    return Path(custom) if custom else Path.home()


def detect_target(
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
    filename: str = "font.ttf",
) -> FontTarget:
    env = os.environ if environ is None else environ
    base = home or user_home(env)
    system = platform or sys.platform
    if is_termux(env):
        return FontTarget(
            "termux", base / ".termux" / "font.ttf", "Termux reads ~/.termux/font.ttf"
        )
    if system.startswith("linux"):
        return FontTarget(
            "linux",
            base / ".local" / "share" / "fonts" / "eurus" / filename,
            "Select the font in your terminal's settings",
        )
    if system == "darwin":
        return FontTarget(
            "macos", base / "Library" / "Fonts" / filename, "Select the font in your terminal"
        )
    return FontTarget(
        "manual", None, "Install the font file by hand, then select it in your terminal"
    )


def is_font_bytes(head: bytes) -> bool:
    return head[:4] in MAGIC


def choose_member(names: list[str], pattern: str) -> str:
    candidates = [
        name
        for name in names
        if not name.endswith("/") and fnmatch.fnmatch(Path(name).name, pattern)
    ]
    if not candidates:
        raise FontError(f"No file matching {pattern} in the archive.")
    return sorted(candidates, key=lambda item: (len(Path(item).name), item))[0]


def extract_font(archive: Path, pattern: str) -> tuple[str, bytes]:
    try:
        with zipfile.ZipFile(archive) as bundle:
            member = choose_member(bundle.namelist(), pattern)
            info = bundle.getinfo(member)
            if info.file_size > MAX_FONT:
                raise FontError("Font file inside the archive is too large.")
            data = bundle.read(member)
    except zipfile.BadZipFile as exc:
        raise FontError("Downloaded file is not a valid zip archive.") from exc
    if not is_font_bytes(data):
        raise FontError("Archive member is not a font file.")
    return Path(member).name, data


def download(url: str, destination: Path, limit: int = MAX_DOWNLOAD) -> None:
    if not url.startswith("https://"):
        raise FontError("Only https downloads are allowed.")
    import httpx

    try:
        with (
            httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=True) as client,
            client.stream("GET", url) as response,
        ):
            if response.url.scheme != "https":
                raise FontError("Download was redirected to a non-https address.")
            if response.status_code != 200:
                raise FontError(f"Download failed with HTTP {response.status_code}.")
            size = 0
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes(65536):
                    size += len(chunk)
                    if size > limit:
                        raise FontError("Download is larger than the allowed limit.")
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        raise FontError(f"Network error: {exc}") from exc


def fetch(spec: FontSpec, fetcher: Callable[[str, Path], None] = download) -> tuple[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="eurus-font-") as folder:
        archive = Path(folder) / "font.zip"
        fetcher(spec.archive, archive)
        return extract_font(archive, spec.pattern)


def read_local(path: Path, pattern: str = "*.ttf") -> tuple[str, bytes]:
    source = path.expanduser()
    if not source.is_file():
        raise FontError(f"File not found: {source}")
    if source.suffix.lower() == ".zip":
        return extract_font(source, pattern)
    if source.suffix.lower() not in (".ttf", ".otf"):
        raise FontError("Use a .ttf, .otf or .zip file.")
    if source.stat().st_size > MAX_FONT:
        raise FontError("Font file is too large.")
    data = source.read_bytes()
    if not is_font_bytes(data):
        raise FontError("File is not a valid font.")
    return source.name, data


def run_command(command: list[str]) -> bool:
    if shutil.which(command[0]) is None:
        return False
    try:
        result = subprocess.run(command, capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def install(
    filename: str,
    data: bytes,
    target: FontTarget,
    runner: Runner = run_command,
) -> InstallResult:
    if target.path is None:
        raise FontError(target.note)
    if not is_font_bytes(data):
        raise FontError("Refusing to install data that is not a font.")
    destination = target.path
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup = destination.with_name(destination.name + BACKUP_SUFFIX)
        if target.kind == "termux" and destination.exists() and not backup.exists():
            shutil.copy2(destination, backup)
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(data)
        partial.replace(destination)
    except OSError as exc:
        raise FontError(f"Cannot write font: {exc}") from exc
    if target.kind == "termux":
        reloaded = runner(["termux-reload-settings"])
        message = (
            "Font applied; Termux reloaded."
            if reloaded
            else "Font installed; run termux-reload-settings or restart Termux."
        )
    elif target.kind == "linux":
        reloaded = runner(["fc-cache", "-f", str(destination.parent)])
        message = f"Font installed as {filename}; select it in your terminal settings."
    else:
        reloaded = False
        message = f"Font installed as {filename}; select it in your terminal settings."
    return InstallResult(destination, reloaded, message)


def install_source(
    source: str,
    environ: Mapping[str, str] | None = None,
    fetcher: Callable[[str, Path], None] = download,
    runner: Runner = run_command,
) -> InstallResult:
    local = Path(source).expanduser()
    if source.strip().lower() not in CATALOG and local.exists():
        name, data = read_local(local)
    else:
        name, data = fetch(lookup(source), fetcher)
    return install(name, data, detect_target(environ, filename=name), runner)


def reset(target: FontTarget, runner: Runner = run_command) -> str:
    if target.kind != "termux" or target.path is None:
        return "Reset only applies to Termux; remove the font in your terminal settings."
    backup = target.path.with_name(target.path.name + BACKUP_SUFFIX)
    try:
        if backup.exists():
            backup.replace(target.path)
            message = "Original Termux font restored."
        elif target.path.exists():
            target.path.unlink()
            message = "Custom font removed; Termux default restored."
        else:
            return "No custom font is installed."
    except OSError as exc:
        raise FontError(f"Cannot restore font: {exc}") from exc
    runner(["termux-reload-settings"])
    return message


def status(target: FontTarget) -> str:
    if target.path is None:
        return target.note
    if target.kind == "termux":
        if target.path.is_file():
            return f"Custom font installed ({target.path.stat().st_size // 1024} KB)."
        return "Using the Termux default font."
    folder = target.path.parent
    installed = sorted(item.name for item in folder.glob("*.[to]tf")) if folder.is_dir() else []
    return f"Installed: {', '.join(installed)}" if installed else "No Eurus fonts installed."


def lookup(key: str) -> FontSpec:
    spec = CATALOG.get(key.strip().lower())
    if spec is None:
        raise FontError(f"Unknown font: {key}. Available: {', '.join(CATALOG)}")
    return spec
