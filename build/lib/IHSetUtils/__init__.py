# src/__init__.py

# Import modules and functions from your package here
from .conversion_IHData import got2ncEBSC, gos2ncEBSC, gow2ncEBSC
from .geometry import abs_angle_cartesian, abs_pos, cartesianDir2nauticalDir, interp_lon, nauticalDir2cartesianDir, nauticalDir2cartesianDirP, pol2cart, rel_angle_cartesian, rel_angle_cartesianP, shore_angle
from .morfology import ADEAN, ALST, BruunRule, depthOfClosure, Hs12Calc, wast, wMOORE, deanSlope
from .waves import BreakingPropagation, GroupCelerity, hunt, LinearShoal, LinearShoalBreak_Residual, LinearShoalBreak_ResidualVOL, RelDisp, RU2_Stockdon2006, Snell_Law