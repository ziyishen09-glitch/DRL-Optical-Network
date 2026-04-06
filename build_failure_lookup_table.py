"""Build a failure-link impact lookup table based on K-shortest paths.

Example:
    python build_failure_lookup_table.py 
        --topology COST239 
        --channels 8 
        --failure-link 1,3 
        --k 3 
        --output results_ppo/failure_lookup_COST239_Failure_1-3_k3.json
"""

from __future__ import annotations

import argparse
from typing import Sequence

from rwa_wdm.net.factory import get_net_instance_from_args
from rwa_wdm.util import (
    build_failure_link_lookup_ksp,
    coerce_link_argument,
    parse_link_argument,
    save_failure_link_lookup,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Precompute failure-link lookup table for SBP acceleration')
    parser.add_argument('--topology', required=True, help='Topology identifier, e.g. COST239_Failure')
    parser.add_argument('--channels', type=int, required=True, help='Number of wavelength channels')
    parser.add_argument('--failure-link', required=True, type=parse_link_argument, help='Failure link as src,dst or src-dst')
    parser.add_argument('--k', type=int, default=3, help='K value for K-shortest-path impact lookup')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    failure_link = coerce_link_argument(args.failure_link)
    if failure_link is None:
        raise ValueError('A failure link must be provided')

    net = get_net_instance_from_args(args.topology, args.channels)
    lookup = build_failure_link_lookup_ksp(net.a, failure_link, args.k)
    save_failure_link_lookup(
        args.output,
        lookup,
        topology=args.topology,
        channels=args.channels,
        failure_link=failure_link,
        k_paths=max(1, int(args.k)),
    )

    print(f'Lookup table saved: {args.output}')
    print(f'Failure link: {failure_link[0]}-{failure_link[1]}')
    print(f'Pairs indexed: {len(lookup)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
