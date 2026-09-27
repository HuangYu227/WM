"""Add an isolated SAP-Bind section to a verified WorldTTT JSON config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import FIVE_BINDING_LAYERS, binding_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--layers', nargs='+', type=int, default=list(FIVE_BINDING_LAYERS))
    args = parser.parse_args()
    settings = binding_settings(json.loads(Path(args.base).read_text(encoding='utf-8')),
                                layers=args.layers)
    Path(args.output).write_text(json.dumps(settings, indent=2), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
