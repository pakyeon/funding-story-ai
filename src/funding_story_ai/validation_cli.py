from __future__ import annotations

import json

from .data_repository import DataRepository


def validation_summary(repository: DataRepository | None = None) -> dict:
    repository = repository or DataRepository()
    repository.check_schemas()
    templates = repository.load_templates()
    repository.validate_catalog_links()
    brief = repository.load_brief()
    return {
        "status": "ok",
        "templates": {
            "count": len(templates),
            "ids": [template["id"] for template in templates],
            "section_counts": {
                template["id"]: len(template["layout"]) for template in templates
            },
        },
        "example": {
            "brief_id": brief["brief_id"],
            "purpose": brief["source"]["purpose"],
            "unknown_count": len(brief["unknowns"]),
        },
    }


def main() -> None:
    print(json.dumps(validation_summary(), ensure_ascii=False))


if __name__ == "__main__":
    main()
