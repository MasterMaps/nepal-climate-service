"""Contract checks for this instance's dataset plugins.

These exist because a plugin class is imported **lazily, at ingest time** — the API
starts fine and the dataset still lists in `/collections` even when its plugin is
broken, so a breakage only surfaces when someone runs that ingest.

That is not hypothetical: bumping the pinned `open-climate-service` release to `main`
silently broke `era5_tmax_monthly`, which imported two private symbols
(`_CdsClient`, `_era5land_probe`) that upstream had deleted. Nothing caught it.

The tests below walk every `ingestion.plugin` path declared in `plugins/datasets/*.yaml`
and assert the class imports and satisfies the current plugin contract, so an
incompatible upstream bump fails here instead of mid-ingest.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from open_climate_service.streaming import BaseDatasetPlugin

_DATASETS_DIR = Path(__file__).resolve().parent.parent / "plugins" / "datasets"


def _declared_plugin_paths() -> list[tuple[str, str]]:
    """Return (template_id, dotted plugin path) for every template that declares one."""
    found: list[tuple[str, str]] = []
    for yaml_path in sorted(_DATASETS_DIR.glob("*.yaml")):
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        templates = loaded if isinstance(loaded, list) else [loaded]
        for template in templates:
            if not isinstance(template, dict):
                continue
            plugin = (template.get("ingestion") or {}).get("plugin")
            if isinstance(plugin, str) and plugin:
                found.append((str(template.get("id", yaml_path.stem)), plugin))
    return found


_PLUGIN_PATHS = _declared_plugin_paths()


def test_templates_declare_plugins() -> None:
    """Guard the guard: if this collapses to nothing, the tests below vacuously pass."""
    assert _PLUGIN_PATHS, f"no ingestion.plugin paths found under {_DATASETS_DIR}"


def _load(path: str) -> Any:
    module_name, _, class_name = path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@pytest.mark.parametrize(("dataset_id", "plugin_path"), _PLUGIN_PATHS, ids=lambda v: str(v))
class TestDeclaredPlugin:
    def test_imports(self, dataset_id: str, plugin_path: str) -> None:
        """The class resolves — catches upstream removing something we import."""
        _load(plugin_path)

    def test_follows_current_contract(self, dataset_id: str, plugin_path: str) -> None:
        """Subclasses BaseDatasetPlugin, per the adding-custom-datasets guide."""
        plugin = _load(plugin_path)
        assert issubclass(plugin, BaseDatasetPlugin), (
            f"{plugin_path} must subclass BaseDatasetPlugin "
            "(see https://dhis2.github.io/open-climate-service/adding_custom_datasets/)"
        )

    def test_implements_the_two_required_methods(self, dataset_id: str, plugin_path: str) -> None:
        plugin = _load(plugin_path)
        for method in ("periods", "fetch_period"):
            assert callable(getattr(plugin, method, None)), f"{plugin_path} must implement {method}()"

    def test_has_no_dead_probe_method(self, dataset_id: str, plugin_path: str) -> None:
        """`probe()` is never called since the grid is inferred from the first period.

        A plugin still defining it is on the retired contract, which fails *silently*:
        the dtype/nodata/CRS it declares via `GridSpec` are ignored, so the store can
        be written with different semantics than the plugin intends.
        """
        plugin = _load(plugin_path)
        assert not hasattr(plugin, "probe"), (
            f"{plugin_path} still defines probe() — retired contract; "
            "delete it and let the orchestrator infer the grid"
        )

    def test_declares_canonical_dimension_names(self, dataset_id: str, plugin_path: str) -> None:
        """The orchestrator reads these off the plugin to find the fetched dims."""
        plugin = _load(plugin_path)
        assert (plugin.time_dim, plugin.y_dim, plugin.x_dim) == ("t", "y", "x")


# --- categorical datasets must coarsen by majority, not by mean -------------------------------

# Templates whose values are class codes rather than measurements. Averaging class codes is not
# merely imprecise: on the ESA CCI scheme, a 2x2 block of 10 (cropland, rainfed) and 50 (tree
# cover, broadleaved evergreen) averages to 20 — "cropland, irrigated", a real class that
# describes neither input. The pyramid the map viewer reads above full zoom is built from that,
# so the wrong answer looks like a legitimate one.
_CATEGORICAL_TEMPLATE_IDS = {"esa_landcover_yearly", "modis_landcover_yearly"}


def _templates() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for yaml_path in sorted(_DATASETS_DIR.glob("*.yaml")):
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        out.extend(t for t in (loaded if isinstance(loaded, list) else [loaded]) if isinstance(t, dict))
    return out


def test_categorical_templates_declare_mode_resampling() -> None:
    """A categorical dataset must declare `ingestion.resampling: mode`.

    The default is `mean`, which is correct for measurements and wrong for class codes. This is
    a template-level decision that nothing in core can infer, so it has to be asserted here.
    """
    by_id = {str(t.get("id")): t for t in _templates()}
    missing = sorted(_CATEGORICAL_TEMPLATE_IDS - set(by_id))
    assert not missing, f"categorical template ids no longer present: {missing} (rename in the test too)"

    for dataset_id in sorted(_CATEGORICAL_TEMPLATE_IDS):
        resampling = (by_id[dataset_id].get("ingestion") or {}).get("resampling")
        assert resampling == "mode", (
            f"{dataset_id} declares ingestion.resampling={resampling!r}; categorical class codes "
            "must coarsen by majority or the pyramid averages them into a different class"
        )


def test_continuous_templates_do_not_declare_mode_resampling() -> None:
    """The converse, so the rule above is not applied by reflex to measurements.

    `clms_ndvi` is the case worth guarding: its `units` are null like the land-cover datasets,
    because NDVI is dimensionless — not because it is categorical. Its mean is meaningful.
    """
    for template in _templates():
        dataset_id = str(template.get("id"))
        if dataset_id in _CATEGORICAL_TEMPLATE_IDS:
            continue
        resampling = (template.get("ingestion") or {}).get("resampling")
        assert resampling != "mode", f"{dataset_id} is not categorical but declares resampling: mode"
