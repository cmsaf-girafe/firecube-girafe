"""Tests for GIRAFE Ingestor"""

from importlib.metadata import entry_points

from firecube.ingestor.api import discover_ingestors

from firecube_girafe import GirafeIngestor


def test_entry_point_registers_the_ingestor() -> None:
    found = entry_points(group="firecube.plugins", name="girafe")
    assert found, "pyproject.toml declares no firecube.plugins entry point named girafe"
    for entry_point in found:
        entry_point.load()
    assert discover_ingestors()["girafe"] is GirafeIngestor
    assert GirafeIngestor.PRODUCT_NAME == "girafe"
