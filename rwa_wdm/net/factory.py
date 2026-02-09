"""Shared helpers for instantiating network topologies."""

from .net import Network

__all__ = ["get_net_instance_from_args"]


def get_net_instance_from_args(topname: str, numch: int) -> Network:
    """Return the topology instance behind the string identifier."""

    if topname == "nsf":
        from .nsf import NationalScienceFoundation

        return NationalScienceFoundation(numch)
    elif topname == "clara":
        from .clara import CooperacionLatinoAmericana

        return CooperacionLatinoAmericana(numch)
    elif topname == "janet":
        from .janet import JointAcademicNetwork

        return JointAcademicNetwork(numch)
    elif topname == "rnp":
        from .rnp import RedeNacionalPesquisa

        return RedeNacionalPesquisa(numch)
    elif topname == "pdf":
        from .topologypdf import MyTopology

        return MyTopology(numch)
    elif topname == "auxgraph_demo_net":
        from .auxgraph_demo_net import auxgraph_demo_net

        return auxgraph_demo_net(numch)
    elif topname == "COST239":
        from .COST239 import COST239

        return COST239(numch)
    else:
        raise ValueError("No network named '%s'" % topname)
