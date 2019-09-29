import math


def idm(v, s, v_leader):
    acce_max = 5
    decel_max = 4
    time_headway = 2
    v0 = 30
    s0 = 30

    acce_free = acce_max * (1 - (v / v0) ** 4)
    if v_leader is not None:
        delta_v = v - v_leader
        s_exp = s0 + v*time_headway + v*delta_v/(2*math.sqrt(acce_max*decel_max))
        acce_int = -acce_max*(s_exp/s)**2
    else:
        acce_int = 0.0
    acce = acce_free + acce_int

    return acce
