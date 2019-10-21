import os
import sys
import math
import random
import datetime
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from IDM import IDM
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


class Road:
    def __init__(self):
        """
        assume all lanes have the same width
        """
        self.entranceEdgeID = 'entranceEdge'
        self.rampExitEdgeID = 'rampExit'
        self.highwayKeepEdgeID = 'exit'

        self.highwayKeepRouteID = 'keep_on_highway'
        self.rampExitRouteID = 'ramp_exit'

        self.entranceEdgeLaneID_0 = self.entranceEdgeID + '_0'
        self.laneNum = traci.edge.getLaneNumber(self.entranceEdgeID)
        self.laneWidth = traci.lane.getWidth(self.entranceEdgeLaneID_0)
        self.laneLength = traci.lane.getLength(self.entranceEdgeLaneID_0)

        self.rampEntranceJunction = traci.junction.getPosition('rampEntrance')
        self.startJunction = list(traci.junction.getPosition('start'))


class Vehicle:
    def __init__(self, veh_id, rd):
        self.veh_id = veh_id
        self.speed = None
        self.latSpeed = 0
        self.latSpeed_last = 0
        self.latAcce = 0
        self.acce = 0
        self.acce_last = self.acce
        self.delta_acce = 0

        self.pos = rd.startJunction
        self.pos_x = self.pos[0]
        self.pos_y = self.pos[1]
        self.laneIndex = traci.vehicle.getLaneIndex(self.veh_id)
        self.pos_lat = traci.vehicle.getLateralLanePosition(self.veh_id) + (self.laneIndex+0.5)*rd.laneWidth
        self.pos_lat_last = self.pos_lat
        self.lanePos = traci.vehicle.getLanePosition(self.veh_id)
        self.yawAngle = 0

        self.leader = None
        self.leaderDis = None
        self.leaderID = None
        self.leaderSpeed = None

        self.targetLeaderID = None
        self.targetFollowerID = None

        self.dis2tgtLane = None
        self.dis2entrance = None
        self.lcPos = None
        self.reward = 0
        # not update every step
        self.targetLane = self.laneIndex
        self.origLane = self.laneIndex

        self.is_ego = 0
        self.changeTimes = 0
        self.idm_obj = None

        traci.vehicle.setLaneChangeMode(veh_id, 256)  # 768

    def setTargetLane(self, tgl):
        self.targetLane = tgl

    def update_info(self, rd, veh_dict):
        self.laneIndex = traci.vehicle.getLaneIndex(self.veh_id)
        self.speed = traci.vehicle.getSpeed(self.veh_id)

        self.acce = traci.vehicle.getAcceleration(self.veh_id)
        self.delta_acce = (self.acce - self.acce_last) / 0.1
        self.acce_last = self.acce

        self.pos_lat_last = self.pos_lat
        self.pos_lat = traci.vehicle.getLateralLanePosition(self.veh_id) + (self.laneIndex+0.5)*rd.laneWidth
        self.latSpeed_last = self.latSpeed
        self.latSpeed = (self.pos_lat - self.pos_lat_last) / 0.1  # 0.1 for time step length
        self.latAcce = (self.latSpeed - self.latSpeed_last) / 0.1

        self.pos = traci.vehicle.getPosition(self.veh_id)
        self.pos_x = self.pos[0]
        self.pos_y = self.pos[1]
        self.lanePos = traci.vehicle.getLanePosition(self.veh_id)
        self.yawAngle = math.atan(self.latSpeed / max(self.speed, 0.00000001))

        self.leader = traci.vehicle.getLeader(self.veh_id)

        if traci.vehicle.getLeader(self.veh_id) is not None:
            if self.leader[0] in list(veh_dict.keys()):
                self.leaderID = self.leader[0]
                self.leaderDis = self.leader[1]
                self.leaderSpeed = traci.vehicle.getSpeed(self.leaderID)
            else:
                self.leaderID = None
                self.leaderDis = None
                self.leaderSpeed = None
        else:
            self.leaderID = None
            self.leaderDis = None
            self.leaderSpeed = None
        # the following 2 all in list [(id1, distance1), (id2, distance2), ...], each tuple is a leader on one lane
        # for a single left lane, the list only contains 1 tuple, eg.[(id1, distance1)]
        # todo modify getNeighbor once changed to target lane, only follow target leader, leader==targetLeader
        if self.laneIndex > self.targetLane:
            if len(traci.vehicle.getNeighbors(self.veh_id, int(self.laneIndex > self.targetLane)+2+0)) != 0:

                self.targetLeaderID = traci.vehicle.getNeighbors(self.veh_id, int(self.laneIndex>self.targetLane)+2+0)[0][0]
                if self.targetLeaderID not in list(veh_dict.keys()):
                    self.targetLeaderID = None
            else:
                self.targetLeaderID = None
            if len(traci.vehicle.getNeighbors(self.veh_id, int(self.laneIndex > self.targetLane)+0+0)) != 0:
                self.targetFollowerID = traci.vehicle.getNeighbors(self.veh_id, int(self.laneIndex>self.targetLane)+0+0)[0][0]
                if self.targetFollowerID not in list(veh_dict.keys()):
                    self.targetFollowerID = None
            else:
                self.targetFollowerID = None
        else:
            assert self.laneIndex == self.targetLane
            self.targetLeaderID = self.leaderID
            self.targetFollowerID = self.leaderID

        self.dis2tgtLane = abs((0.5 + self.targetLane) * rd.laneWidth - self.pos_lat)
        self.dis2entrance = rd.laneLength - self.lanePos
        # self.dis2entrance = self.calculate_dis(self.pos_x, self.pos_y, enrtrance_x, entrance_y)

    def updateLongitudinalSpeedIDM(self):
        """
        use IDM to control vehicle speed
        :return:
        """
        # cannot acquire vNext, compute longitudinal speed on our own
        if self.leaderID is not None:
            acceNext = self.idm_obj.calc_acce(self.speed, self.leaderDis, self.leaderSpeed)
        else:
            acceNext = self.idm_obj.calc_acce(self.speed, None, None)

        return acceNext

    def calcAcce(self):
        if self.leaderID is None and self.targetLeaderID is None:
            print('no leader')
            acceNext = self.idm_obj.calc_acce(self.speed, None, None)
        elif self.leaderID is None and self.targetLeaderID is not None:
            print('following target leader')
            acceNext = self.idm_obj.calc_acce(self.speed, traci.vehicle.getLanePosition(self.targetLeaderID) - self.lanePos, traci.vehicle.getSpeed(self.targetLeaderID))
        elif self.leaderID is not None and self.targetLeaderID is None:
            print('following current leader')
            acceNext = self.idm_obj.calc_acce(self.speed, self.leaderDis, self.leaderSpeed)
        else:
            assert self.leaderID is not None and self.targetLeaderID is not None
            if self.leaderDis > (traci.vehicle.getLanePosition(self.targetLeaderID) - self.lanePos):
                print('following closer leader: TARGET:%s' % self.targetLeaderID)
                acceNext = self.idm_obj.calc_acce(self.speed,
                                                  traci.vehicle.getLanePosition(self.targetLeaderID) - self.lanePos,
                                                  traci.vehicle.getSpeed(self.targetLeaderID))
            else:
                print('following closer leader: CURRENT LANE:%s' % self.leaderID)
                acceNext = self.idm_obj.calc_acce(self.speed, self.leaderDis, self.leaderSpeed)
        return acceNext

    def changeLane(self, cps, tgtlane, rd):
        # make compulsory/default lane change, do not respect other vehicles
        '''
        if tgtlane == 0:
            traci.vehicle.setRouteID(self.veh_id, rd.rampExitRouteID)
        else:
            traci.vehicle.setRouteID(self.veh_id, rd.highwayKeepRouteID)
        assert traci.vehicle.isRouteValid(self.veh_id) is True, 'route is not valid'
        '''
        # set lane change mode
        if tgtlane != -1:
            if cps is True:
                traci.vehicle.setLaneChangeMode(self.veh_id, 0)
                # execute lane change with 'changeSublane'
            else:
                traci.vehicle.setLaneChangeMode(self.veh_id, 1621)  # 768:no speed adaption
                # traci.vehicle.changeLane(self.veh_id, self.targetLane, 1)
            traci.vehicle.changeSublane(self.veh_id, (0.5 + tgtlane) * rd.laneWidth - self.pos_lat)
        else:
            traci.vehicle.changeSublane(self.veh_id, 0.0)

        if self.dis2tgtLane < 0.1:

            return True
        else:
            return False

    def calculate_dis(self, x, y, x0, y0):
        """
        calculate distance between 2 positions
        """
        return math.sqrt((x-x0)**2 + (y-y0)**2)


class LaneChangeEnv(gym.Env):
    def __init__(self, id=None, traffic=1, gui=False, seed=None):
        # todo check traffic flow density
        if traffic == 0:
            # average 9 vehicles
            self.cfg = '/Users/cxx/Desktop/lcEnv/map/ramp3/mapFree.sumo.cfg'
        elif traffic == 2:
            # average 19 vehicles
            self.cfg = '/Users/cxx/Desktop/lcEnv/map/ramp3/mapDense.sumo.cfg'
        else:
            # average 14 vehicles
            self.cfg = '/Users/cxx/Desktop/lcEnv/map/ramp3/map.sumo.cfg'

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

        # gym env return values
        # todo observation type Box --completed
        '''
        self.observation = {'leader': {'exist': 0,
                                       'speed': 0,
                                       'pos': 0},
                            'rightLeader': {'exist': 0,
                                             'speed': 0,
                                             'pos': 0},
                            'rightFollower': {'exist': 0,
                                               'speed': 0,
                                               'pos': 0}
                            }         # (object): agent's observation of the current environment'''
        self.observation = [[0, 0, 0],  # ego lane position and speed
                            [0, 0, 0],  # leader
                            [0, 0, 0],  # target lane leader
                            [0, 0, 0]]  # target lane follower
        self.reward = None            # (float) : amount of reward returned after previous action
        self.done = True              # (bool): whether the episode has ended, in which case further step() calls will return undefined results
        self.info = {'resetFlag': 0}  # (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3))
        '''
        self.observation_space = spaces.Dict({'leader': spaces.Dict({'exist': spaces.Discrete(2),
                                                                     'speed': spaces.Box(-np.inf, np.inf, shape=(1,)),
                                                                     'pos': spaces.Box(-np.inf, np.inf, shape=(1,))}),
                                              'rightLeader': spaces.Dict({'exist': spaces.Discrete(2),
                                                               'speed': spaces.Box(-np.inf, np.inf, shape=(1,)),
                                                               'pos': spaces.Box(-np.inf, np.inf, shape=(1,))}),
                                              'rightFollower': spaces.Dict({'exist': spaces.Discrete(2),
                                                                 'speed': spaces.Box(-np.inf, np.inf, shape=(1,)),
                                                                 'pos': spaces.Box(-np.inf, np.inf, shape=(1,))})
                                              })'''

    def update_veh_dict(self, veh_id_tuple):
        for veh_id in veh_id_tuple:
            if veh_id not in self.veh_dict.keys():
                self.veh_dict[veh_id] = Vehicle(veh_id, self.rd)

        for veh_id in list(self.veh_dict.keys()):
            if veh_id not in veh_id_tuple:
                self.veh_dict.pop(veh_id)

        for veh_id in list(self.veh_dict.keys()):
            self.veh_dict[veh_id].update_info(self.rd, self.veh_dict)

    def _updateObservationSingle(self, name, id):
        """
        :param name: 0:ego; 1:leader; 2:target leader; 3:target follower
        :param id: vehicle id corresponding to name
        :return:
        """
        if id is not None:
            self.observation[name][0] = traci.vehicle.getLanePosition(id)
            self.observation[name][1] = traci.vehicle.getSpeed(id)
            self.observation[name][2] = self.veh_dict[id].pos_lat
        else:
            self.observation[name][0] = self.observation[0][0] + 300.
            self.observation[name][1] = self.observation[0][1]
            self.observation[name][2] = 4.8
            # todo check if rational

    def updateObservation(self, egoid):
        self._updateObservationSingle(0, egoid)
        self._updateObservationSingle(1, self.ego.leaderID)
        #print(self.ego.targetLeaderID)
        self._updateObservationSingle(2, self.ego.targetLeaderID)
        self._updateObservationSingle(3, self.ego.targetFollowerID)

    def updateReward(self):
        r_total = -abs(self.ego.dis2tgtLane)
        return r_total

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
            print('lateralPos2leader', abs(self.ego.pos_lat - self.veh_dict[self.ego.leaderID].pos_lat))
            alpha = abs(self.ego.pos_lat - self.veh_dict[self.ego.leaderID].pos_lat) / 3.2
            assert 0 <= alpha <= 1.1
            r_safe_leader = w_lateral*alpha + w_longi*(1-alpha)*abs(self.ego.leaderDis)
        else:
            r_safe_leader = 0
        if self.ego.targetLeaderID is not None:
            print('lateralPos2tgtleader', abs(self.ego.pos_lat - self.veh_dict[self.ego.targetLeaderID].pos_lat))
            alpha = abs(self.ego.pos_lat - self.veh_dict[self.ego.targetLeaderID].pos_lat) / 3.2
            print('alpha', alpha)
            assert 0 <= alpha <= 1.1

            r_safe_tgtleader = w_lateral*alpha + w_longi*(1-alpha)*abs(self.ego.lanePos - self.veh_dict[self.ego.targetLeaderID].lanePos)
        else:
            r_safe_tgtleader = 0
        r_safe = r_safe_leader + r_safe_tgtleader

        # total reward
        r_total = r_comf + r_effi_all + r_safe

        return r_total

    def is_done(self):
        # lane change successfully executed, episode ends, reset env
        # todo modify
        if self.is_success:
            self.done = True
            print('reset on: successfully lane change, dis2targetlane:',
                  self.ego.dis2tgtLane)
        # too close to ramp entrance
        if self.ego.dis2entrance < 10.0:
            self.done = True
            print('reset on: too close to ramp entrance, dis2targetlane:',
                  self.ego.dis2tgtLane)
        # ego vehicle out of env
        if self.egoID not in self.vehID_tuple_all:
            self.done = True
            print('reset on:', '\nself.ego not in env:', self.egoID not in self.vehID_tuple_all)
        # collision occurs
        self.collision_num = traci.simulation.getCollidingVehiclesNumber()
        if self.collision_num > 0:
            self.done = True
            print('\nself.collision_num:', self.collision_num)

    def preStep(self):
        traci.simulationStep()
        self.vehID_tuple_all = traci.edge.getLastStepVehicleIDs(self.rd.entranceEdgeID)
        self.update_veh_dict(self.vehID_tuple_all)

    def IDMStep(self):
        # using idm to control longitudinal dynamics
        assert self.egoID is not None
        if self.egoID is not None:
            acceNext = self.ego.updateLongitudinalSpeedIDM()
            vNext = self.ego.speed + acceNext*0.1
            traci.vehicle.setSpeed(self.egoID, vNext)

        traci.simulationStep()
        self.vehID_tuple_all = traci.edge.getLastStepVehicleIDs(self.rd.entranceEdgeID)
        self.update_veh_dict(self.vehID_tuple_all)
        # todo simplify network

    def decision(self):
        # only use self.observation
        # lateral action -1:decelerate; 1:accelerate; 0: keep
        if self.observation[1][0] != np.inf:
            dis2leader = self.observation[1][0] - self.observation[0][0]
            if dis2leader > 23:
                act_longi = 1
            elif dis2leader < 20:
                act_longi = -1
            else:
                act_longi = 0
        else:
            act_longi = 0

        act_lateral = 2
        return act_longi, act_lateral

    def clipVelocity(self, v):
        if v < 15:
            vClipped = 15
        elif v > 35:
            vClipped = 35
        else:
            vClipped = v
        return vClipped

    def longiCtrl(self, action_longi):
        if action_longi == 1:
            acce = 5
        elif action_longi == -1:
            acce = -4
        else:
            acce = 0

        vNextCmd = self.ego.speed + acce * 0.1
        vNext = self.clipVelocity(vNextCmd)

        traci.vehicle.setSpeedMode(self.egoID, 0)
        traci.vehicle.setSpeed(self.egoID, vNext)

    def step(self, action=(1, 2)):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, call `reset()` outside env!! to reset this
        environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): longitudinal: action[0] = 1: accelerate
                                           action[0] = -1: decelerate
                                           action[0] = 0: use SUMO default
                                           action[0] = others: acce = 0.0

                                           action[0] = 0: follow leader in current lane
                                           action[0] = 1: follow argmin(dis2leader in current lane, dis2leader in target lane)
                             lateral: action[1] = 1: lane change
                                      action[1] = 0: abort lane change, change back to original lane
                                      action[1] = 2: keep in current lateral position

        Returns:
            described in __init__
        """
        action_longi = action[0]
        action_lateral = action[1]

        assert self.done is False, 'self.done is not False'
        assert action is not None, 'action is None'
        assert self.egoID in self.vehID_tuple_all, 'vehicle not in env'

        self.timestep += 1

        # lateral control-------------------------
        # episode in progress; 0:change back to original line; 1:lane change to target lane; 2:keep current
        # lane change to target lane
        if not self.is_success:
            if action_lateral == 1: #and abs(self.ego.pos_lat - (0.5+self.ego.targetLane)*self.rd.laneWidth) > 0.01:
                self.is_success = self.ego.changeLane(True, self.ego.targetLane, self.rd)
                print('posLat', self.ego.pos_lat, 'lane', self.ego.laneIndex, 'rdWdith', self.rd.laneWidth)
                print('right', -(self.ego.pos_lat - 0.5*self.rd.laneWidth))
            # abort lane change, change back to ego's original lane
            if action_lateral == 0: #and abs(self.ego.pos_lat - (0.5+self.ego.origLane)*self.rd.laneWidth) > 0.01:
                self.is_success = self.ego.changeLane(True, self.ego.origLane, self.rd)
                print('left', 1.5 * self.rd.laneWidth - self.ego.pos_lat)
            #  keep current lateral position
            if action_lateral == 2:
                self.is_success = self.ego.changeLane(True, -1, self.rd)


        # longitudinal control---------------------
        '''choice 1 of longitudinal actions
        if action_longi != 0:
            self.longiCtrl(action_longi)
        '''
        if action_longi == 0:
            # follow leader in current lane
            acceNext = self.ego.updateLongitudinalSpeedIDM()
        else:
            print('action_longi', action_longi)
            assert action_longi == 1, 'action_longi invalid!'
            acceNext = self.ego.calcAcce()

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
            self.updateObservation(self.egoID)
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
            self.ego.targetLane = tlane
            self.ego.is_ego = 1

            self.ego_speedFactor = traci.vehicle.getSpeedFactor(egoid)
            self.ego_speedLimit = self.ego_speedFactor * traci.lane.getMaxSpeed(traci.vehicle.getLaneID(self.egoID))

            self.ego.idm_obj = IDM()
            self.ego.idm_obj.__init__(self.ego_speedLimit)
            self.ego.update_info(self.rd, self.veh_dict)
            self.updateObservation(self.egoID)
            #self.step(2)  # todo
            return self.observation
        return

