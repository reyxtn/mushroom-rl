from .locomotion import HopperWarp, Walker2DWarp, AntWarp, HalfCheetahWarp
from .locomotion.unitree.go2 import Go2Base, Go2Walk

Go2Base.register()
Go2Walk.register()
HopperWarp.register()
Walker2DWarp.register()
AntWarp.register()
HalfCheetahWarp.register()
