import pytest

from eurus.agent.prompt import build_system_prompt
from eurus.config import load_config
from eurus.skills import SkillError, list_skills, load_skill, resolve_skills


def test_builtin_antislop_is_discoverable(tmp_path):
    skills = list_skills(tmp_path)
    assert any(skill.name == "antislop" for skill in skills)


def test_builtin_antislop_loads_content(tmp_path):
    skill = load_skill("antislop", tmp_path)
    assert skill.source == "builtin"
    assert skill.description
    assert len(skill.content) > 1000


def test_skill_prompt_uses_index_and_tiers(tmp_path):
    prompt = build_system_prompt(tmp_path, skills=("antislop",))
    assert "<eurus-tier:stable>" in prompt
    assert "<eurus-tier:context>" in prompt
    assert "<eurus-tier:volatile>" in prompt
    assert "Skill index:" in prompt
    assert "<eurus-skill" not in prompt


def test_no_skill_prompt_when_disabled(tmp_path):
    prompt = build_system_prompt(tmp_path, skills=())
    assert "<eurus-skill" not in prompt


def test_workspace_skill_overrides_builtin(tmp_path):
    skill_dir = tmp_path / ".eurus" / "skills" / "antislop"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: antislop\ndescription: workspace override\n---\n"
        "Workspace-specific instructions.",
        encoding="utf-8",
    )
    skill = load_skill("antislop", tmp_path)
    assert skill.source == "workspace"
    assert skill.content == "Workspace-specific instructions."


def test_invalid_skill_name_rejected():
    with pytest.raises(SkillError, match="invalid skill name"):
        load_skill("../secret")


def test_malformed_workspace_skill_is_not_discovered(tmp_path):
    skill_dir = tmp_path / "skills" / "badskill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("no frontmatter", encoding="utf-8")
    assert all(skill.name != "badskill" for skill in list_skills(tmp_path))
    with pytest.raises(SkillError):
        load_skill("badskill", tmp_path)


def test_config_defaults_to_antislop(tmp_path):
    config = load_config(cwd=tmp_path, environ={"EURUS_HOME": str(tmp_path / "home")})
    assert config.agent.skills == ()


def test_config_can_disable_skills(tmp_path):
    config_path = tmp_path / "eurus.toml"
    config_path.write_text("[agent]\nskills = []\n", encoding="utf-8")
    config = load_config(
        config_path=config_path,
        cwd=tmp_path,
        environ={"EURUS_HOME": str(tmp_path / "home")},
    )
    assert config.agent.skills == ()


def test_config_skills_env_override(tmp_path):
    config = load_config(
        cwd=tmp_path,
        environ={"EURUS_HOME": str(tmp_path / "home"), "EURUS_SKILLS": "antislop,custom"},
    )
    assert config.agent.skills == ("antislop", "custom")


def test_config_rejects_invalid_skill_name(tmp_path):
    config_path = tmp_path / "eurus.toml"
    config_path.write_text('[agent]\nskills = ["../oops"]\n', encoding="utf-8")
    with pytest.raises(Exception, match="invalid skill name"):
        load_config(
            config_path=config_path,
            cwd=tmp_path,
            environ={"EURUS_HOME": str(tmp_path / "home")},
        )


def test_resolve_skills_deduplicates_in_order(tmp_path):
    selected = resolve_skills(("antislop", "antislop"), tmp_path)
    assert [skill.name for skill in selected] == ["antislop"]


def test_system_prompt_contains_skill_index_without_skill_body(tmp_path):
    prompt = build_system_prompt(tmp_path, skills=())
    assert "Skill index:" in prompt
    assert "antislop —" in prompt
    assert "Supplemental skill instructions" not in prompt


def test_skill_view_tool_loads_requested_skill(tmp_path):
    import asyncio
    import json

    from eurus.skills.tools import skill_tools
    from eurus.tools.base import ToolContext

    tool = skill_tools()[0]
    result = asyncio.run(tool.handler({"name": "antislop"}, ToolContext(workspace=tmp_path)))
    payload = json.loads(result)
    assert payload["name"] == "antislop"
    assert "content" in payload
    assert len(payload["content"]) > 1000
