import math


def idm(v, s, v_leader):
    acce_max = 5
    decel_max = 4
    time_headway = 2
    v0 = 50
    s0 = 50

    delta_v = v - v_leader
    s_exp = s0 + v*time_headway + v*delta_v/(2*math.sqrt(acce_max*decel_max))
    acce_free = acce_max*(1 - (v/v0)**4)
    acce_int = -acce_max*(s_exp/s)**2
    acce = acce_free + acce_int

    return acce
