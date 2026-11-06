from .locomotion import HopperWarp, Walker2DWarp, AntWarp, HalfCheetahWarp
from .locomotion.unitree.go2 import Go2Base, Go2Walk
from .locomotion.unitree.h2 import H2Base, H2Walk, H2Stand

Go2Base.register()
Go2Walk.register()
H2Base.register()
H2Walk.register()
H2Stand.register()

HopperWarp.register()
Walker2DWarp.register()
AntWarp.register()
HalfCheetahWarp.register()
