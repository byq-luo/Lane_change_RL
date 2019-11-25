import math


class IDM():
    def __init__(self, v0=39.44):
        self.acce_max = 2.9
        self.decel_max = 4.5
        # todo change time headway
        self.time_headway = 1.0
        self.v0 = v0
        self.s0 = 2.5

    def setSpeedLimit(self, new_sl):
        self.v0 = new_sl

    def calc_acce(self, v, s, v_leader):
        acce_free = self.acce_max * (1 - (v / self.v0) ** 4)
        if v_leader is not None:
            delta_v = v - v_leader
            s_exp = self.s0 + v*self.time_headway + v*delta_v/(2*math.sqrt(self.acce_max*self.decel_max))
            acce_int = -self.acce_max*(s_exp/s)**2
        else:
            acce_int = 0.0
        acce = acce_free + acce_int

        return acce
