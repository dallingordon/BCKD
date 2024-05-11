from ._base import Vanilla
from .KD import KD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .SLD import SLD #dg added 1/30
from .SLDONLY import SLDONLY
from .SLDR import SLDR
from .SLDM import SLDM
from .SLDE import SLDE
from .KDS import KDS
from .SLDMSE import SLDMSE
from .SLDMSEAVG import SLDMSEAVG
from .BD import BD
from .BDAVG import BDAVG
from .BDMSEV import BDMSEV
from .BDMSE import BDMSE
from .BCP import BCP
from .BCPLI import BCPLI
from .BLDMSE import BLDMSE
from .BLDCD import BLDCD
from .BLDMSEP import BLDMSEP
from .BLDCDP import BLDCDP
from .BLDCDPMP import BLDCDPMP
from .BLDCD2P import BLDCD2P
from .SLDMSEPROD import SLDMSEPROD
from .SLDMSE2PROD import SLDMSE2PROD
from .MACDM import MACDM
from .MACDP import MACDP
from .MACDML import MACDML
from .MACDPL import MACDPL
from .MACDMM import MACDMM
from .MACDAI import MACDAI
from .MIX_I import MIX_I
from .MIX_II import MIX_II
from .MIX_III import MIX_III
from .MIX_IV import MIX_IV
from .MIX_V import MIX_V
from .MIX_VI import MIX_VI
from .MIX_VI_MAX import MIX_VI_MAX
from .MIX_VI_R import MIX_VI_R

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "SLD": SLD,
    "SLDONLY": SLDONLY,
    "SLDR": SLDR,
    "SLDM": SLDM,
    "SLDE": SLDE,
    "SLDMSE": SLDMSE,
    "KDS": KDS,
    "BD": BD,
    "SLDMSEAVG": SLDMSEAVG,
    "BDAVG": BDAVG,
    "BDMSEV": BDMSEV,
    "BDMSE": BDMSE,
    "BCP": BCP,
    "BCPLI": BCPLI,
    "BLDMSE": BLDMSE,
    "BLDCD": BLDCD,
    "BLDMSEP": BLDMSEP,
    "BLDCDP": BLDCDP,
    "BLDCDPMP": BLDCDPMP,
    "BLDCD2P": BLDCD2P,
    "SLDMSEPROD": SLDMSEPROD,
    "SLDMSE2PROD": SLDMSE2PROD,
    "MACDM": MACDM,
    "MACDP": MACDP,
    "MACDML": MACDML,
    "MACDPL": MACDPL,
    "MACDMM": MACDMM,
    "MACDAI": MACDAI,
    "MIX_I": MIX_I,
    "MIX_II": MIX_II,
    "MIX_III": MIX_III,
    "MIX_IV": MIX_IV,
    "MIX_V": MIX_V,
    "MIX_VI": MIX_VI,
    "MIX_VI_MAX": MIX_VI_MAX,
    "MIX_VI_R": MIX_VI_R,
    
    
    
}
