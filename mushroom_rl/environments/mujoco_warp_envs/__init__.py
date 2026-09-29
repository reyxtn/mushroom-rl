from .locomotion import HopperWarp, Walker2DWarp, AntWarp, HalfCheetahWarp

from .go2 import Go2Base, Go2Walk
from .h2 import H2Base, H2Stand, H2Walk, H2Run

Go2Base.register()
Go2Walk.register()
H2Base.register()
H2Stand.register()
H2Walk.register()
H2Run.register()
HopperWarp.register()
Walker2DWarp.register()
AntWarp.register()
HalfCheetahWarp.register()
