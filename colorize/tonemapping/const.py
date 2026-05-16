from scipy.interpolate import make_splrep

# Credit - https://rawpedia.rawtherapee.com/Color_Management#DCP_Tone_Curve DCP_tone_curve.rcp
BASE_CURVE_ADOBE = make_splrep([0,0.1,0.32,0.66,1],
                               [0,0.09,0.43,0.87,1],
                               k=3)