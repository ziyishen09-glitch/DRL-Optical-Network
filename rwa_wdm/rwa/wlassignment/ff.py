"""First-fit wavelength assignment strategy

"""
from typing import List, Optional

# FIXME https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
from ...net import Network


def first_fit(net: Network, route: List[int], contains_virtual: bool = False, enable_new_ff: bool = False) -> Optional[List[int]]:
    """First-fit algorithm

    Select the wavelength with the lowest index available at the first link of
    the path, starting of course from the source node.

    Args:
        net: Network object
        route: path encoded as a sequence of router indices

    Returns:
        :obj:`list[int]` or ``None``: upon wavelength assignment success, return
            a list of wavelength indices to be used on the lightpath. The list
            contains one wavelength index per link in the route (for the
            per-link first-fit mode) or a single-element list when a single
            wavelength is used across the whole route. Returns ``None`` on
            failure.

    """
    w_list = []
    # Sanity: route must contain at least one hop
    if not route or len(route) < 2:
        return None
    
    if not enable_new_ff:
        # try a single wavelength across the whole route (first-fit)
        for w in range(net.nchannels):
            ok = True
            w_list_candidate = []
            consumed_keys: list[tuple[int, int]] = []
            # iterate pairwise over route to get links
            for idx in range(len(route) - 1):
                i, j = route[idx], route[idx + 1]
                if net.n[i][j][w]:
                    # wavelength available on this link
                    w_list_candidate.append(w)
                else:
                    # wavelength not available: try consuming 10 QKP keys for this link
                    # (each request occupies 10 time-slots in the data layer)
                    try:
                        used = net.use_qkp((i, j), 10)
                    except Exception:
                        used = False
                    if used:
                        # mark this link as satisfied by QKP (use sentinel -10)
                        w_list_candidate.append(-10)
                        consumed_keys.append((i, j))
                    else:
                        # cannot satisfy this link with this wavelength
                        ok = False
                        break
            if ok:
                return w_list_candidate
            else:
                # rollback any consumed QKP keys for this candidate wavelength
                if consumed_keys:
                    for (ii, jj) in consumed_keys:
                        try:
                            net.add_qkp((ii, jj), 10)
                        except Exception:
                            # best-effort rollback; ignore failures
                            pass
        return None
    
    else:
        consumed_keys: list[tuple[int, int]] = []
        for idx in range(len(route) - 1):
            i, j = route[idx], route[idx + 1]
            ok = False
            for w in range(net.nchannels):
                if net.n[i][j][w]:
                    ok = True
                    w_list.append(w)
                    break
            if not ok:
                # try to consume 10 QKP keys for this link
                try:
                    used = net.use_qkp((i, j), 10)
                except Exception:
                    used = False
                if used:
                    w_list.append(-10)
                    consumed_keys.append((i, j))
                    ok = True
                else:
                    # rollback any previously consumed keys on failure
                    if consumed_keys:
                        for (ii, jj) in consumed_keys:
                            try:
                                net.add_qkp((ii, jj), 10)
                            except Exception:
                                pass
                    return None
        return w_list