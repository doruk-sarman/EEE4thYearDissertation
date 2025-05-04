import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------- CONFIG ---------------------------------- #
PV_KWP          = 3.0              # rated PV power (kW_p)
WIND_KW_RATED   = 5.0              # rated wind-turbine power (kW)
ETA_CONV        = 0.92             # inverter / wiring efficiency (η_conv)
G_STC           = 1000.0           # irradiance at STC (W m-2)

# wind-speed break-points for the power curve (m s-1)
CUT_IN, RATED, CUT_OUT = 3.0, 12.0, 25.0   

# ----------------------- 1.  LOAD ERA5 SERIES -------------------------- #
era = (pd.read_csv("era5_hourly.csv", parse_dates=["timestamp"])
         .set_index("timestamp")
         .sort_index())

# ----------------------- 2.  PV GENERATION ----------------------------- #
# Linear model:  P_PV = kWp · (GHI / G_STC), clipped at nameplate
p_pv_kw = PV_KWP * (era["ghi_w_m2"] / G_STC)
p_pv_kw = p_pv_kw.clip(upper=PV_KWP)

# ----------------------- 3.  WIND GENERATION --------------------------- #
def turbine_power(v_ms: pd.Series) -> pd.Series:
    """Piece-wise cubic power curve."""
    p = np.where(
        v_ms < CUT_IN, 0.0,
        np.where(
            v_ms < RATED,
            WIND_KW_RATED * ((v_ms - CUT_IN) / (RATED - CUT_IN)) ** 3,
            np.where(v_ms < CUT_OUT, WIND_KW_RATED, 0.0)
        )
    )
    return pd.Series(p, index=v_ms.index)

p_wind_kw = turbine_power(era["wind_speed_ms"])

# ----------------------- 4.  COMBINE & APPLY η ------------------------- #
p_re_kw = ETA_CONV * (p_pv_kw + p_wind_kw)

out = pd.DataFrame({
    "p_pv_kw"  : p_pv_kw.round(3),
    "p_wind_kw": p_wind_kw.round(3),
    "p_re_kw"  : p_re_kw.round(3)
}, index=era.index)

out.to_csv("re_profile.csv", index_label="timestamp")
print("Finished. Combined profile written to re_profile.csv")