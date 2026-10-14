from axiom import paths


def test_home_dir_uses_axiom_home(tmp_path):
    assert paths.home_dir({"AXIOM_HOME": str(tmp_path)}) == tmp_path


def test_user_config_path(tmp_path):
    assert paths.user_config_path({"AXIOM_HOME": str(tmp_path)}) == tmp_path / "config.toml"


def test_project_config_prefers_root_file(tmp_path):
    (tmp_path / ".axiom").mkdir()
    (tmp_path / ".axiom" / "config.toml").write_text("")
    (tmp_path / "axiom.toml").write_text("")
    assert paths.project_config_path(tmp_path) == tmp_path / "axiom.toml"


def test_project_config_missing(tmp_path):
    assert paths.project_config_path(tmp_path) is None
