# Copyright (C) 2026 Masahiro Hiramori
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import torch
import yaml

from nets.tuss import TussModel


@torch.no_grad()
def test_tuss_export_avoids_aliasing_complex_input_views():
    config_path = Path(__file__).resolve().parents[1] / "pretrained_models/tuss.medium.2-4src/hparams.yaml"
    config = yaml.safe_load(config_path.read_text())
    model = TussModel(**config["model_conf"]).eval()
    input = torch.randn(1, 8, 1025, dtype=torch.complex64)
    prompts = [["speech", "sfxbg", "musicbg"]]

    # Populate RoPE caches before capture, as their initialization mutates buffers.
    expected = model(input, prompts)
    exported = torch.export.export(model, (input, prompts), strict=False).run_decompositions()
    targets = {node.target for node in exported.graph.nodes if node.op == "call_function"}

    assert torch.ops.aten.view_as_real_copy.default in targets
    assert torch.ops.aten.view_as_real.default not in targets
    torch.testing.assert_close(exported.module()(input, prompts), expected, rtol=1e-4, atol=1e-5)
