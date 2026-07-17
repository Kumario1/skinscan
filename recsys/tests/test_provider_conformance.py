import hashlib
import json
from pathlib import Path

from recsys.catalog import CatalogProduct
from recsys.contracts import Profile
from recsys.knowledge import load_knowledge
from recsys.scoring import score_products
from recsys.signals import ScoringContext, SignalScore, load_providers


DATA = Path(__file__).parents[1] / "data"


def _product():
    return CatalogProduct(
        product_id="p1", name="Test", brand="Test", category="serum",
        price_usd=10, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=(),
    )


def _context():
    return ScoringContext(
        targets=(), profile=Profile(),
        knowledge=load_knowledge(DATA / "knowledge"), category_prices={},
    )


def test_new_registry_provider_needs_no_scorer_or_composer_changes(tmp_path, monkeypatch):
    from recsys import signals

    class DummyProvider:
        name = "dummy"

        def __init__(self, store, meta):
            self.version = meta["version"]

        def score(self, product, slot, ctx):
            return SignalScore(self.name, 0.9, "dummy evidence")

    data_root = tmp_path / "data"
    store_path = data_root / "signals" / "dummy.v1.json"
    store_path.parent.mkdir(parents=True)
    store_path.write_text('{}\n')
    digest = hashlib.sha256(store_path.read_bytes()).hexdigest()
    (data_root / "signals" / "registry.json").write_text(json.dumps({
        "schema_version": "recsys-registry-1",
        "stores": [{
            "name": "dummy", "kind": "dummy", "version": "v1",
            "path": "signals/dummy.v1.json", "sha256": digest,
            "source": {"catalog_sha256": "catalog-1"}, "status": "active",
        }],
    }))
    monkeypatch.setitem(signals.STORE_PROVIDERS, "dummy", DummyProvider)

    providers, _meta, warnings = load_providers(data_root, "catalog-1")
    scored = score_products([_product()], "serum", providers, _context(), {"dummy": 1})
    assert warnings == []
    assert scored[0].final == 0.9
