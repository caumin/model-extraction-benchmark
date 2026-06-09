"""Attack implementations."""

from mebench.attackers.runner import AttackRunner
from mebench.attackers.activethief import ActiveThief
from mebench.attackers.dfme import DFME
from mebench.attackers.ds import DualStudents
from mebench.attackers.maze import MAZE
from mebench.attackers.dfms import DFMSHL
from mebench.attackers.disguide import DisGUIDE
from mebench.attackers.game import GAME
from mebench.attackers.es_attack import ESAttack
from mebench.attackers.swiftthief import SwiftThief
from mebench.attackers.blackbox_dissector import BlackboxDissector
from mebench.attackers.cloudleak import CloudLeak
from mebench.attackers.blackbox_ripper import BlackboxRipper
from mebench.attackers.marich import MARICH
from mebench.attackers.copycatcnn import CopycatCNN
from mebench.attackers.inversenet import InverseNet
from mebench.attackers.knockoff_nets import KnockoffNets
from mebench.attackers.random_baseline import RandomBaseline

# Backwards-compatible alias
DFMS = DFMSHL

__all__ = [
    "AttackRunner",
    "ActiveThief",
    "DFME",
    "DualStudents",
    "MAZE",
    "DFMSHL",
    "DFMS",
    "DisGUIDE",
    "GAME",
    "ESAttack",
    "SwiftThief",
    "BlackboxDissector",
    "CloudLeak",
    "BlackboxRipper",
    "MARICH",
    "CopycatCNN",
    "InverseNet",
    "KnockoffNets",
    "RandomBaseline",
]
