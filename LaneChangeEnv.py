import os
import sys
import math
import random
import datetime
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
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
        self.latSpeed = None
        self.acce = None
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

        self.targetLeaderID = None
        self.targetFollowerID = None

        self.dis2entrance = None
        self.lcPos = None
        self.reward = 0
        # not update every step
        self.targetLane = self.laneIndex
        self.origLane = self.laneIndex

        self.is_ego = 0
        self.change_times = 0

        traci.vehicle.setLaneChangeMode(veh_id, 512)

    def setTargetLane(self, tgl):
        self.targetLane = tgl

    def update_info(self, rd):
        # use subscriptions instead
        self.laneIndex = traci.vehicle.getLaneIndex(self.veh_id)
        self.speed = traci.vehicle.getSpeed(self.veh_id)
        self.acce = traci.vehicle.getAcceleration(self.veh_id)
        self.pos_lat_last = self.pos_lat

        self.pos_lat = traci.vehicle.getLateralLanePosition(self.veh_id) + (self.laneIndex+0.5)*rd.laneWidth
        #print('LateralLanePosition', traci.vehicle.getLateralLanePosition(self.veh_id))
        #print('laneIndex', self.laneIndex)
        self.pos = traci.vehicle.getPosition(self.veh_id)
        self.pos_x = self.pos[0]
        self.pos_y = self.pos[1]
        self.lanePos = traci.vehicle.getLanePosition(self.veh_id)
        self.latSpeed = (self.pos_lat - self.pos_lat_last) / 0.1  # 0.1 for time step length
        self.yawAngle = math.atan(self.latSpeed / max(self.speed, 0.00000001))

        self.leader = traci.vehicle.getLeader(self.veh_id)

        if traci.vehicle.getLeader(self.veh_id) is not None:
            self.leaderID = self.leader[0]
            self.leaderDis = self.leader[1]
        else:
            self.leaderID = None
            self.leaderDis = None
        # the following 2 all in list [(id1, distance1), (id2, distance2), ...], each tuple is a leader on one lane
        # for a single left lane, the list only contains 1 tuple, eg.[(id1, distance1)]
        if len(traci.vehicle.getNeighbors(self.veh_id, int(self.laneIndex > self.targetLane)+2+0)) != 0:
            self.targetLeaderID = traci.vehicle.getNeighbors(self.veh_id, int(self.laneIndex>self.targetLane)+2+0)[0][0]
        else:
            self.targetLeaderID = None
        if len(traci.vehicle.getNeighbors(self.veh_id, int(self.laneIndex > self.targetLane)+0+0)) != 0:
            self.targetFollowerID = traci.vehicle.getNeighbors(self.veh_id, int(self.laneIndex>self.targetLane)+0+0)[0][0]
        else:
            self.targetFollowerID = None

        self.dis2entrance = rd.laneLength - self.lanePos
        # self.dis2entrance = self.calculate_dis(self.pos_x, self.pos_y, enrtrance_x, entrance_y)
        if self.is_ego == 1:
            self.update_reward()

    def changeLane(self, cps, tgtlane, rd):
        # make compulsory/default lane change, do not respect other vehicles
        if tgtlane == 0:
            traci.vehicle.setRouteID(self.veh_id, rd.rampExitRouteID)
        else:
            traci.vehicle.setRouteID(self.veh_id, rd.highwayKeepRouteID)
        assert traci.vehicle.isRouteValid(self.veh_id) is True, 'route is not valid'
        # set lane change mode
        if cps is True:
            traci.vehicle.setLaneChangeMode(self.veh_id, 0)
            # execute lane change with 'changeSublane'
            traci.vehicle.changeSublane(self.veh_id, (0.5 + self.targetLane) * rd.laneWidth - self.pos_lat)
        else:
            traci.vehicle.setLaneChangeMode(self.veh_id, 512)
            # traci.vehicle.changeLane(self.veh_id, self.targetLane, 1)
            traci.vehicle.changeSublane(self.veh_id, (0.5 + self.targetLane) * rd.laneWidth - self.pos_lat)

    def update_reward(self):
        # todo define reward
        self.reward = -abs(self.dis2entrance - 100)

    def calculate_dis(self, x, y, x0, y0):
        """
        calculate distance between 2 positions
        """
        return math.sqrt((x-x0)**2 + (y-y0)**2)


class LaneChangeEnv(gym.Env):
    def __init__(self, id=None, traffic=1, gui=True, seed=None):
        # todo check traffic flow density
        if traffic == 0:
            # average 9 vehicles
            self.cfg = '/Users/cxx/Desktop/SUMO/map/ramp3/mapFree.sumo.cfg'
        elif traffic == 2:
            # average 19 vehicles
            self.cfg = '/Users/cxx/Desktop/SUMO/map/ramp3/mapDense.sumo.cfg'
        else:
            # average 14 vehicles
            self.cfg = '/Users/cxx/Desktop/SUMO/map/ramp3/map.sumo.cfg'

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

        self.veh_dict = {}
        self.vehID_tuple_all = ()

        self.egoID = id
        self.ego = None
        # self.tgtLane = tgtlane

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
        self.observation = [[0, 0],  # ego lane position and speed
                            [0, 0],  # leader
                            [0, 0],  # target lane leader
                            [0, 0]]  # target lane follower
        self.reward = None            # (float) : amount of reward returned after previous action
        self.done = True              # (bool): whether the episode has ended, in which case further step() calls will return undefined results
        self.info = {'resetFlag': 0}  # (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, 4))
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
            else:
                self.veh_dict[veh_id].update_info(self.rd)

    def _updateObservationSingle(self, name, id):
        """
        :param name: 0:ego; 1:leader; 2:target leader; 3:target follower
        :param id: vehicle id corresponding to name
        :return:
        """
        if id is not None:
            self.observation[name][0] = traci.vehicle.getLanePosition(id)
            self.observation[name][1] = traci.vehicle.getSpeed(id)
        else:
            self.observation[name][0] = np.inf
            self.observation[name][1] = np.inf
            # todo check if rational
        '''
        self.observation[name]['exist'] = 1
        self.observation[name]['speed'] = traci.vehicle.getSpeed(id)
        self.observation[name]['pos'] = traci.vehicle.getLanePosition(id)
    '''

    def updateObservation(self, egoid):
        self._updateObservationSingle(0, egoid)
        self._updateObservationSingle(1, self.ego.leaderID)
        self._updateObservationSingle(2, self.ego.targetLeaderID)
        self._updateObservationSingle(3, self.ego.targetFollowerID)

    def is_done(self):
        # lane change successfully executed, episode ends, reset env
        if abs(self.ego.pos_lat - (0.5 + self.ego.targetLane) * self.rd.laneWidth) <= 0.01:
            self.done = True
            print('reset on: successfully lane change, dis2targetlane:',
                  self.ego.pos_lat - (0.5 + self.ego.targetLane) * self.rd.laneWidth)
        # too close to ramp entrance
        if self.ego.dis2entrance < 10.0:
            self.done = True
            print('reset on: too close to ramp entrance, dis2targetlane:',
                  self.ego.pos_lat - (0.5 + self.ego.targetLane) * self.rd.laneWidth)
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

    def step(self, action=None):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, call `reset()` outside env!! to reset this
        environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            described in __init__
        """
        assert self.done is False, 'self.done is not False'
        assert action is not None, 'action is None'
        assert self.egoID in self.vehID_tuple_all, 'vehicle not in env'

        self.timestep += 1

        # episode in progress; 0:change back to original line; 1:lane change to target lane; 2:keep cureent
        # lane change to target lane
        if action == 1 and abs(self.ego.pos_lat - (0.5+self.ego.targetLane)*self.rd.laneWidth) > 0.01:

            self.ego.changeLane(True, self.ego.targetLane, self.rd)
            print('posLat', self.ego.pos_lat, 'lane', self.ego.laneIndex, 'rdWdith', self.rd.laneWidth)
            print('right', -(self.ego.pos_lat - 0.5*self.rd.laneWidth))
        # abort lane change, change back to ego's original lane
        if action == 0 and abs(self.ego.pos_lat - (0.5+self.ego.origLane)*self.rd.laneWidth) > 0.01:
            self.ego.changeLane(True, self.ego.origLane, self.rd)
            print('left', 1.5 * self.rd.laneWidth - self.ego.pos_lat)
        #  keep current lateral position
        if action == 2:
            traci.vehicle.changeSublane(self.egoID, 0.0)

        # todo check where simulation step and vhe_dict should be placed
        # update info
        traci.simulationStep()
        self.vehID_tuple_all = traci.edge.getLastStepVehicleIDs(self.rd.entranceEdgeID)
        self.update_veh_dict(self.vehID_tuple_all)
        # check episode ends or not
        self.is_done()
        if self.done is True:
            self.info['resetFlag'] = True
            return self.observation, 0.0, self.done, self.info
        else:
            self.updateObservation(self.egoID)
            self.reward = self.ego.reward
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
        traci.close()
        self.__init__(id=egoid, traffic=tfc, gui=is_gui, seed=sumoseed)
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
            self.updateObservation(self.egoID)
            #self.step(2)  # todo
            return self.observation
        return

