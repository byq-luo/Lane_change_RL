import os
import sys
import random
import datetime
import gym
from gym import spaces
import numpy as np
from env.IDM import IDM
from env.Road import Road
from env.Vehicle import Vehicle

import math
# add sumo/tools into python environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


######################################################################
# simulation environments


class LaneChangeEnv(gym.Env):
    def __init__(self, id=None, traffic=1, gui=False, seed=None):
        # todo check traffic flow density
        if traffic == 0:
            # average 9 vehicles
            self.cfg = 'C:/Users/Fei Ye/Desktop/map/ramp3/mapFree.sumo.cfg'
        elif traffic == 2:
            # average 19 vehicles
            self.cfg = 'C:/Users/Fei Ye/Desktop/map/ramp3/mapDense.sumo.cfg'
        else:
            # average 14 vehicles
            self.cfg = 'C:/Users/Fei Ye/Desktop/map/ramp3/map.sumo.cfg'

        # arguments must be string, if float/int, must be converted to str(float/int), instead of '3.0'
        self.sumoBinary = "/usr/local/Cellar/sumo/1.2.0/bin/sumo"
        self.sumoCmd = ['-c', self.cfg,
                        # '--lanechange.duration', str(3),     # using 'Simple Continuous lane-change model'
                        '--lateral-resolution', str(0.8),  # using 'Sublane-Model'
                        '--step-length', str(0.1),
                        '--default.action-step-length', str(0.1)]
        # randomness
        if seed is None:
            self.sumoCmd += ['--random']
        else:
            self.sumoCmd += ['--seed', str(seed)]
        # gui
        if gui is True:
            self.sumoBinary += '-gui'
            self.sumoCmd = [self.sumoBinary] + self.sumoCmd + ['--quit-on-end', str(True),
                                                               '--start', str(True)]
        else:
            self.sumoCmd = [self.sumoBinary] + self.sumoCmd

        traci.start(self.sumoCmd)

        self.rd = Road()
        self.timestep = 0
        self.dt = traci.simulation.getDeltaT()
        self.randomseed = None
        self.sumoseed = None

        self.veh_dict = {}
        self.vehID_tuple_all = ()

        self.egoID = id
        self.ego = None
        # self.tgtLane = tgtlane
        self.is_success = False

        self.collision_num = 0
        self.lateral_action = 2
        # self.observation = [[0, 0, 0],  # ego lane position and speed
        #                     [0, 0, 0],  # leader
        #                     [0, 0, 0],  # target lane leader
        #                     [0, 0, 0]]  # target lane follower
        self.observation = np.empty(20)
        self.reward = None  # (float) : amount of reward returned after previous action
        self.done = True  # (bool): whether the episode has ended, in which case further step() calls will return undefined results
        self.info = {
            'resetFlag': 0}  # (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,))

    def update_veh_dict(self, veh_id_tuple):
        for veh_id in veh_id_tuple:
            if veh_id not in self.veh_dict.keys():
                self.veh_dict[veh_id] = Vehicle(veh_id, self.rd)

        for veh_id in list(self.veh_dict.keys()):
            if veh_id not in veh_id_tuple:
                self.veh_dict.pop(veh_id)

        for veh_id in list(self.veh_dict.keys()):
            self.veh_dict[veh_id].update_info(self.rd, self.veh_dict)

    def _updateObservationSingle(self, name, veh):
        """
        :param name: 0:ego; 1:leader; 2:target leader; 3:target follower
        :param id: vehicle id corresponding to name
        :return:
        """
        if veh is not None:
            self.observation[name * 4 + 0] = veh.lanePos
            self.observation[name * 4 + 1] = veh.speed
            self.observation[name * 4 + 2] = veh.pos_lat
            self.observation[name * 4 + 3] = veh.acce
        else:
            self.observation[name * 4 + 0] = self.observation[0] + 300.
            self.observation[name * 4 + 1] = self.observation[1]
            self.observation[name * 4 + 2] = 4.8
            self.observation[name * 4 + 3] = 0
            # todo check if rational

    def updateObservation(self):
        self.observation[0] = self.ego.lanePos
        self.observation[1] = self.ego.speed
        self.observation[2] = self.ego.pos_lat
        self.observation[3] = self.ego.acce

        self._updateObservationSingle(1, self.ego.orig_leader)
        self._updateObservationSingle(2, self.ego.orig_follower)
        self._updateObservationSingle(3, self.ego.trgt_leader)
        self._updateObservationSingle(4, self.ego.trgt_follower)
        # self.observation = np.array(self.observation).flatten()
        # print(self.observation.shape)

    def updateReward(self):
        return -self.ego.dis2tgtLane

    def updateReward2(self):
        wc1 = 1
        wc2 = 1
        wt = 1
        ws = 1
        we = 1
        # reward related to comfort
        r_comf = wc1 * self.ego.acce ** 2 + wc2 * self.ego.delta_acce ** 2

        # reward related to efficiency
        r_time = - wt * self.timestep
        r_speed = ws * (self.ego.speed - self.ego_speedLimit)
        r_effi = we * self.ego.dis2tgtLane / self.ego.dis2entrance
        r_effi_all = r_time + r_speed + r_effi

        # reward related to safety
        w_lateral = 1
        w_longi = 1

        if self.ego.leaderID is not None:
            # compute longitudinal time gap
            delta_V = self.veh_dict[self.ego.leaderID].speed - self.ego.speed
            delta_A = self.veh_dict[self.ego.leaderID].acce - self.ego.acce

            if delta_A == 0:
                TTC = - abs(self.ego.leaderDis)/delta_V
            else:
                TTC = -delta_V - math.sqrt(delta_V**2 + 2*delta_A * self.ego.leaderDis)
                TTC = TTC/delta_A


            if self.lateral_action != 1 and 0 < TTC < 2:
                r_long_c = - math.exp(-2*TTC+5)
            else:
                r_long_c = 0

            if self.lateral_action == 0: #abort lane change
                alpha = abs(self.ego.pos_lat - self.veh_dict[self.ego.leaderID].pos_lat) / 3.2
                assert 0 <= alpha <= 1.1
                r_lat_c = -math.exp(-4*alpha+5)
            else:
                r_lat_c = 0



        if self.ego.targetLeaderID is not None:

            # compute longitudinal time gap
            delta_V2 = self.veh_dict[self.ego.targetLeaderID].speed - self.ego.speed
            delta_A2 = self.veh_dict[self.ego.targetLeaderID].acce - self.ego.acce
            delta_D2 = self.veh_dict[self.ego.targetLeaderID].lanePos - self.ego.lanePos
            if delta_A2 == 0:
                TTC2 = - abs(delta_D2) / delta_V2
            else:
                TTC2 = -delta_V2 - math.sqrt(delta_V2 ** 2 + 2 * delta_A2 * delta_D2)
                TTC2 = TTC2 / delta_A2

            if self.lateral_action == 1 and 0 < TTC2 < 2:
                r_long_t = - math.exp(-2 * TTC2 + 5)
            else:
                r_long_t = 0

            if self.lateral_action == 1: # lane change
                alpha = abs(self.ego.pos_lat - self.veh_dict[self.ego.targetLeaderID].pos_lat) / 3.2
                assert 0 <= alpha <= 1.1
                r_lat_t = -math.exp(-4*alpha+5)
            else:
                r_lat_t = 0

        r_safe = w_lateral * (r_lat_c + r_lat_t) + w_longi * (r_long_c+ r_long_t)

        #
        # if self.ego.leaderID is not None:
        #     # ('lateralPos2leader', abs(self.ego.pos_lat - self.veh_dict[self.ego.leaderID].pos_lat))
        #     alpha = abs(self.ego.pos_lat - self.veh_dict[self.ego.leaderID].pos_lat) / 3.2
        #     assert 0 <= alpha <= 1.1
        #     r_safe_leader = w_lateral * alpha + w_longi * (1 - alpha) * abs(self.ego.leaderDis)
        # else:
        #     r_safe_leader = 0
        # if self.ego.targetLeaderID is not None:
        #     # print('lateralPos2tgtleader', abs(self.ego.pos_lat - self.veh_dict[self.ego.targetLeaderID].pos_lat))
        #     alpha = abs(self.ego.pos_lat - self.veh_dict[self.ego.targetLeaderID].pos_lat) / 3.2
        #     # print('alpha', alpha)
        #     assert 0 <= alpha <= 1.1
        #
        #     r_safe_tgtleader = w_lateral * alpha + w_longi * (1 - alpha) * abs(
        #         self.ego.lanePos - self.veh_dict[self.ego.targetLeaderID].lanePos)
        # else:
        #     r_safe_tgtleader = 0
        #
        #
        # r_safe = r_safe_leader + r_safe_tgtleader

        # total reward
        r_total = r_comf + r_effi_all + r_safe

        return r_total

    def is_done(self):
        # lane change successfully executed, episode ends, reset env
        # todo modify
        if self.is_success:
            self.done = True
            # print('reset on: successfully lane change, dis2targetlane:',
            #       self.ego.dis2tgtLane)
        # too close to ramp entrance
        if self.ego.dis2entrance < 10.0:
            self.done = True
            # print('reset on: too close to ramp entrance, dis2targetlane:',
            #       self.ego.dis2tgtLane)
        # ego vehicle out of env
        if self.egoID not in self.vehID_tuple_all:
            self.done = True
            # print('reset on: self.ego not in env:', self.egoID not in self.vehID_tuple_all)
        # collision occurs
        self.collision_num = traci.simulation.getCollidingVehiclesNumber()
        if self.collision_num > 0:
            self.done = True
            # print('reset on: self.collision_num:', self.collision_num)

    def preStep(self):
        traci.simulationStep()
        self.vehID_tuple_all = traci.edge.getLastStepVehicleIDs(self.rd.entranceEdgeID)
        self.update_veh_dict(self.vehID_tuple_all)

    def step(self, action=2):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, call `reset()` outside env!! to reset this
        environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): longitudinal0: action[0] = 1: accelerate
                                            action[0] = -1: decelerate
                                            action[0] = 0: use SUMO default
                                            action[0] = others: acce = 0.0

                             longitudinal1: action[0] = 0: follow original lane leader
                                            action[0] = 1: follow closer leader

                             longitudinal2: action[0] = 0: follow original lane leader
                                            action[0] = 1: follow target lane leader

                             **important**: orginal/target lane leader will not change despite the lateral position of
                                            the ego may change

                             lateral: action[1] = 1: lane change
                                      action[1] = 0: abort lane change, change back to original lane
                                      action[1] = 2: keep in current lateral position

        Returns:
            described in __init__
        """

        action_longi = action // 3
        action_lateral = action % 3

        self.lateral_action = action_lateral
        # action_longi = action[0]
        # action_lateral = action[1]

        assert self.done is False, 'self.done is not False'
        assert action is not None, 'action is None'
        assert self.egoID in self.vehID_tuple_all, 'vehicle not in env'

        self.timestep += 1

        # lateral control-------------------------
        # episode in progress; 0:change back to original line; 1:lane change to target lane; 2:keep current
        # lane change to target lane
        if not self.is_success:
            if action_lateral == 1:  # and abs(self.ego.pos_lat - (0.5+self.ego.targetLane)*self.rd.laneWidth) > 0.01:
                self.is_success = self.ego.changeLane(True, self.ego.trgt_laneIndex, self.rd)
                # print('posLat', self.ego.pos_lat, 'lane', self.ego.curr_laneIndex, 'rdWdith', self.rd.laneWidth)
                # print('right', -(self.ego.pos_lat - 0.5*self.rd.laneWidth))
            # abort lane change, change back to ego's original lane
            if action_lateral == 0:  # and abs(self.ego.pos_lat - (0.5+self.ego.origLane)*self.rd.laneWidth) > 0.01:
                self.is_success = self.ego.changeLane(True, self.ego.orig_laneIndex, self.rd)
                # print('left', 1.5 * self.rd.laneWidth - self.ego.pos_lat)
            #  keep current lateral position
            if action_lateral == 2:
                self.is_success = self.ego.changeLane(True, -1, self.rd)

        # longitudinal control2---------------------
        acceNext = self.ego.updateLongitudinalSpeedIDM(action_longi)
        # print(acceNext)
        vNext = self.ego.speed + acceNext * 0.1
        traci.vehicle.setSpeed(self.egoID, vNext)

        # update info------------------------------
        traci.simulationStep()
        self.vehID_tuple_all = traci.edge.getLastStepVehicleIDs(self.rd.entranceEdgeID)
        self.update_veh_dict(self.vehID_tuple_all)
        # check if episode ends
        self.is_done()
        if self.done is True:
            self.info['resetFlag'] = True
            return self.observation, 0.0, self.done, self.info
        else:
            self.updateObservation()
            self.reward = self.updateReward()
            return self.observation, self.reward, self.done, self.info

    def seed(self, seed=None):
        if seed is None:
            self.randomseed = datetime.datetime.now().microsecond
        else:
            self.randomseed = seed
        random.seed(self.randomseed)

    def reset(self, egoid, tlane=0, tfc=1, is_gui=True, sumoseed=None, randomseed=None):
        """
        reset env
        :param id: ego vehicle id
        :param tfc: int. 0:light; 1:medium; 2:dense
        :return: initial observation
        """
        self.seed(randomseed)
        if sumoseed is None:
            self.sumoseed = self.randomseed

        traci.close()
        self.__init__(id=egoid, traffic=tfc, gui=is_gui, seed=self.sumoseed)
        # continue step until ego appears in env
        if self.egoID is not None:
            while self.egoID not in self.veh_dict.keys():
                # must ensure safety in preStpe
                self.preStep()
                if self.timestep > 5000:
                    raise Exception('cannot find ego after 5000 timesteps')

            assert self.egoID in self.vehID_tuple_all, "cannot start training while ego is not in env"
            self.done = False
            self.ego = self.veh_dict[self.egoID]
            self.ego.trgt_laneIndex = tlane
            self.ego.is_ego = 1
            # set ego vehicle speed mode
            traci.vehicle.setSpeedMode(self.ego.veh_id, 0)
            self.ego_speedFactor = traci.vehicle.getSpeedFactor(egoid)
            self.ego_speedLimit = self.ego_speedFactor * traci.lane.getMaxSpeed(traci.vehicle.getLaneID(self.egoID))

            self.ego.idm_obj = IDM()
            self.ego.idm_obj.__init__(self.ego_speedLimit)
            self.ego.update_info(self.rd, self.veh_dict)
            self.updateObservation()
            return self.observation
        return

    def close(self):
        traci.close()
