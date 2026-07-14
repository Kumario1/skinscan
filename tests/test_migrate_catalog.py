import json

from src.recommendation.migrate_catalog import migrate_catalog_v2


def test_storage_migration_preserves_unknowns_without_granting_roles(tmp_path):
    source = tmp_path / "legacy.json"
    destination = tmp_path / "v2.json"
    source.write_text(json.dumps([{
        "product_id": "legacy", "name": "Legacy Cream", "brand": "Brand",
        "category": "moisturizer", "routine_roles": ["moisturizer"],
        "format": "cream", "cadence": "daily", "cadence_source": "legacy://claim",
    }]))
    assert migrate_catalog_v2(source, destination) == {"products": 1, "migrated": 1}
    row = json.loads(destination.read_text())[0]
    assert row["catalog_schema_version"] == "2"
    assert row["routine_roles"] == []
    assert row["format"] == "unknown"
    assert row["cadence"] is None
