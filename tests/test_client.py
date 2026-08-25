from funding_story_ai.client import StoryGenerator
from funding_story_ai.data_repository import DataRepository


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, brief, *, template_id=None):
        self.calls.append((brief, template_id))
        return {"brief_id": brief["brief_id"], "template_id": template_id}


def test_public_client_loads_a_brief_path_and_forwards_options() -> None:
    repository = DataRepository()
    pipeline = FakePipeline()
    generator = StoryGenerator(repository, pipeline)  # type: ignore[arg-type]

    result = generator.generate(
        repository.examples_dir / "robot-vacuum" / "brief.json",
        template="t04_full_campaign",
    )

    assert result == {
        "brief_id": "cleanforge-r1-synthetic-v1",
        "template_id": "t04_full_campaign",
    }
    assert pipeline.calls[0][1] == "t04_full_campaign"
