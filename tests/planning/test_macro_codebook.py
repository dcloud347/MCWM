import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mcwm.actions.schema import ActionSource, CanonicalActionTick
from mcwm.planning.macro_codebook import (
    BASIC_CODE_NAMES,
    MacroCodebook,
        MacroCodebookFitConfig,
        fit_macro_codebook_from_episodes,
        resample_actions_to_model_ticks,
)

try:
    import torch
    from mcwm.planning.legality import LegalityContext, expand_macro_codes, legal_code_mask
except ModuleNotFoundError:
    torch = None


def _action(
    timestamp,
    *,
    movement=(),
    interaction=(),
    hotbar=0,
    camera=(0.0, 0.0),
    gui_open=False,
    valid=True,
):
    movement_names = ("forward", "back", "left", "right", "jump", "sneak", "sprint")
    interaction_names = (
        "attack", "use", "drop", "pick_item", "swap_hands", "inventory", "esc"
    )
    if not valid:
        return CanonicalActionTick.noop(
            timestamp, ActionSource.VPT, valid=False
        )
    return CanonicalActionTick(
        movement=tuple(name in movement for name in movement_names),
        interaction=tuple(name in interaction for name in interaction_names),
        hotbar=hotbar,
        camera=camera,
        cursor=(0.5, 0.5) if gui_open else None,
        gui_open=gui_open,
        valid=True,
        timestamp_ms=timestamp,
        source=ActionSource.VPT,
    )


def _config():
    return MacroCodebookFitConfig(
        min_group_samples=1,
        min_cluster_samples=1,
        max_clusters_per_group=2,
        max_codes=32,
        max_samples_per_group=100,
        seed=17,
    )


class MacroCodebookTest(unittest.TestCase):
    def test_fit_is_deterministic_and_preserves_basic_codes(self):
        actions = (
            _action(0),
            _action(50, movement=("forward",), camera=(0.0, 0.5)),
            _action(100, movement=("forward",), camera=(0.0, 1.0)),
            _action(150, interaction=("attack",)),
            _action(200, interaction=("use",)),
            _action(250, gui_open=True),
        )
        first = fit_macro_codebook_from_episodes(
            (("episode-a", actions),), manifest_hash="manifest", config=_config()
        )
        second = fit_macro_codebook_from_episodes(
            (("episode-a", actions),), manifest_hash="manifest", config=_config()
        )

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertTrue(set(BASIC_CODE_NAMES).issubset({code.name for code in first.codes}))
        self.assertEqual(first.provenance["split"], "train")
        self.assertEqual(first.provenance["source"], "vpt")

    def test_json_round_trip_keeps_hash_and_provenance(self):
        codebook = fit_macro_codebook_from_episodes(
            (("episode-a", (_action(0), _action(50))),),
            manifest_hash="abc123",
            config=_config(),
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "macro_codebook.json"
            codebook.write(path)
            loaded = MacroCodebook.read(path)
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded.to_dict(), codebook.to_dict())
        self.assertEqual(loaded.content_hash, codebook.content_hash)
        self.assertEqual(raw["manifest_hash"], "abc123")
        self.assertEqual(raw["random_seed"], 17)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_legality_masks_conflicts_gui_and_unavailable_hotbar(self):
        actions = (
            _action(0, movement=("forward", "back")),
            _action(50, movement=("forward", "back")),
            _action(100, hotbar=7),
            _action(150, hotbar=7),
            _action(200, gui_open=True),
            _action(250, gui_open=True),
        )
        codebook = fit_macro_codebook_from_episodes(
            (("episode-a", actions),), manifest_hash="m", config=_config()
        )
        mask = legal_code_mask(
            codebook,
            LegalityContext(valid_hotbar_slots=(1, 2, 3)),
        )

        for code in codebook.codes:
            if (
                "mutually_exclusive_forward_back" in code.legality.reasons
                or code.gui_mode != "gameplay"
                or 7 in code.hotbar
            ):
                self.assertFalse(bool(mask[code.code_id]))

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_expansion_produces_eight_model_steps_at_source_rate(self):
        codebook = fit_macro_codebook_from_episodes(
            (), manifest_hash="m", config=_config()
        )
        names = {code.name: code.code_id for code in codebook.codes}
        ids = torch.tensor(
            [[names["forward"], names["turn_right"], names["jump"], names["attack"]]]
        )
        residuals = torch.zeros(1, 8, 2)
        residuals[0, 0, 1] = 0.5
        expanded = expand_macro_codes(codebook, ids, residuals)

        self.assertEqual(tuple(expanded.movement.shape), (1, 8, 5, 7))
        self.assertEqual(tuple(expanded.camera.shape), (1, 8, 5, 2))
        self.assertTrue(bool(expanded.movement[0, 0, 0, 0]))
        self.assertAlmostEqual(float(expanded.camera[0, 0, :, 1].sum()), 0.5)
        self.assertTrue(bool(expanded.valid_mask.all()))

    def test_source_actions_are_aggregated_to_four_fps_model_ticks(self):
        source = tuple(
            _action(
                index * 50,
                movement=("forward",),
                hotbar=3 if index == 2 else 0,
                camera=(0.0, 1.0),
            )
            for index in range(10)
        )
        model_ticks = resample_actions_to_model_ticks(source, _config())

        self.assertEqual(len(model_ticks), 2)
        self.assertEqual([tick.timestamp_ms for tick in model_ticks], [0, 250])
        self.assertEqual(model_ticks[0].hotbar, 3)
        self.assertEqual(model_ticks[1].hotbar, 0)
        self.assertEqual(model_ticks[0].camera, (0.0, 5.0))
        self.assertTrue(model_ticks[0].movement_value("forward"))


if __name__ == "__main__":
    unittest.main()
