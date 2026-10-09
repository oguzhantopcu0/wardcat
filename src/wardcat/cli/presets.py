"""``wardcat presets``: what each starting policy enables, and what it leaves out."""

from __future__ import annotations

import argparse


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "presets", help="list the presets, or show what one enables", allow_abbrev=False
    )
    p.add_argument("name", nargs="?", help="a preset to show in full, e.g. kvkk")
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    from wardcat.cli import EXIT_CLEAN
    from wardcat.presets import PRESETS, get_preset

    if not args.name:
        width = max(len(name) for name in PRESETS)
        for name in sorted(PRESETS):
            print(f"{name:<{width}}  {get_preset(name).covers}")
        return EXIT_CLEAN
    preset = get_preset(args.name)  # an unknown name raises ConfigError (exit 2)
    print(f"{preset.name}: {preset.covers}")
    print(f"not covered: {preset.not_covered}")
    if preset.needs_layers:
        print(f"needs layers: {', '.join(sorted(preset.needs_layers))} for some of its types")
    print()
    width = max(len(entity) for entity in preset.entities)
    for entity, action in sorted(preset.entities.items()):
        print(f"  {entity:<{width}}  {action}")
    return EXIT_CLEAN
