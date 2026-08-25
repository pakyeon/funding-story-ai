from funding_story_ai.data_repository import DataRepository
from funding_story_ai.selector import TemplateSelector


def test_cleanforge_brief_selects_problem_solution_automation_template() -> None:
    repository = DataRepository()
    brief = repository.load_brief("robot-vacuum/brief.json")
    selection = TemplateSelector().select(brief, repository.load_templates())

    assert selection.template_id == "t02_problem_solution_automation"
    assert selection.scores[selection.template_id] == max(selection.scores.values())
    assert "automation/problem signals" in " ".join(selection.reasons)


def test_rule_selector_does_not_depend_on_a_category_profile() -> None:
    repository = DataRepository()
    brief = repository.load_brief("robot-vacuum/brief.json")
    baseline = TemplateSelector().select(brief, repository.load_templates())
    repeated = TemplateSelector().select(brief, repository.load_templates())
    assert repeated == baseline
