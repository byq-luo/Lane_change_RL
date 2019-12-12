import os, sys, random, datetime, gym, math
from gym import spaces
import numpy as np
from env.Road import Road
from env.Vehicle import Vehicle
from env.Ego import Ego
# add sumo/tools into python environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
######################################################################
# lane change environment


class LaneChangeEnv(gym.Env):
    def __init__(self, id=None, traffic=1, gui=False, seed=None):
        # todo check traffic flow density
        if traffic == 0:
            # average 9 vehicles
            #self.cfg = '/Users/cxx/Desktop/lcEnv/map/ramp3/mapFree.sumo.cfg'
            self.cfg = '../map/ramp3/mapFree.sumo.cfg'
        elif traffic == 2:
            # average 19 vehicles
            self.cfg = '../map/ramp3/mapDense.sumo.cfg'
        else:
            # average 14 vehicles
            self.cfg = '../map/ramp3/map.sumo.cfg'

        # arguments must be string, if float/int, must be converted to str(float/int), instead of '3.0'
        self.sumoBinary = "/usr/local/Cellar/sumo/1.2.0/bin/sumo"
        self.sumoCmd = ['-c', self.cfg,
                        # '--lanechange.duration', str(3),     # using 'Simple Continuous lane-change model'
                        '--lateral-resolution', str(0.8),  # using 'Sublane-Model'
                        '--step-length', str(0.1),
                        '--default.action-step-length', str(0.1)]
                        #'--no-warnings']
        # randomness
        if seed is None:
            self.sumoCmd += ['--random']  # initialize with current system time
        else:
            self.sumoCmd += ['--seed', str(seed)]  # initialize with given seed
        # gui
        if gui is True:
            self.sumoBinary += '-gui'
            self.sumoCmd = [self.sumoBinary] + self.sumoCmd + ['--quit-on-end', str(True),
                                                               '--start', str(True)]
        else:
            self.sumoCmd = [self.sumoBinary] + self.sumoCmd
        # start Traci
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

        self.is_success = False
        self.success_timer = 0
        self.collision_num = 0

        self.observation = np.empty(21)
        self.reward = None            # (float) : amount of reward returned after previous action
        self.done = True              # (bool): whether the episode has ended, in which case further step() calls will return undefined results
        self.info = {'resetFlag': 0}  # (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        self.is_done_info = 0
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21, ))

    def update_veh_dict(self, veh_id_tuple):
        for veh_id in veh_id_tuple:
            if veh_id not in self.veh_dict.keys():
                if veh_id == self.egoID:
                    self.veh_dict[veh_id] = Ego(veh_id, self.rd)
                else:
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
        # todo difference
        if veh is not None:
            self.observation[name*4+0+1] = veh.pos_longi - self.observation[0]
            self.observation[name*4+1+1] = veh.speed
            self.observation[name*4+2+1] = veh.pos_lat
            self.observation[name*4+3+1] = veh.acce
        else:
            assert name != self.egoID
            self.observation[name*4+0+1] = 300.
            self.observation[name*4+1+1] = self.observation[1]
            if name == 1 or name == 2:
                self.observation[name*4+2+1] = 4.8
            else:
                self.observation[name*4+2+1] = 1.6
            self.observation[name*4+3+1] = 0
            # todo check if rational

    def updateObservation(self):
        self.observation[0] = self.ego.pos_longi
        self.observation[1] = self.ego.speed
        self.observation[2] = self.ego.pos_lat
        self.observation[3] = self.ego.acce
        self.observation[4] = self.ego.speed_lat

        #self._updateObservationSingle(0, self.ego)
        self._updateObservationSingle(1, self.ego.orig_leader)
        self._updateObservationSingle(2, self.ego.orig_follower)
        self._updateObservationSingle(3, self.ego.trgt_leader)
        self._updateObservationSingle(4, self.ego.trgt_follower)
        # self.observation = np.array(self.observation).flatten()
        # print(self.observation.shape)

    def updateReward(self, action_lateral):
        if self.is_done_info == 0:  # weights
            w_c = 0.005
            w_e = 0.1
            w_s = 10

            # reward related to comfort
            w_a = 1
            w_da = 1
            r_comf = w_c * (- w_a * abs(self.ego.acce) - w_da * abs(self.ego.delta_acce))
            # reward related to efficiency
            w_t = 0.1
            w_sp = 0.1
            w_lc = 1
            r_time = - w_t * self.timestep
            r_speed = - w_sp * abs(self.ego.speed - self.ego.speedLimit)
            r_lc = - w_lc * self.ego.dis2tgtLane
            r_effi = w_e * (r_time + r_speed + r_lc)
            # reward related to safety

            # def _get_safety_reward(ego, veh_temp):
            #     if veh_temp is not None:
            #         dis_longi = abs(veh_temp.pos_longi - ego.pos_longi)
            #         if dis_longi < 12:
            #             # todo consider velocity
            #             return -1/(dis_longi + 0.1)
            #         else:
            #             return 0.0
            #     else:
            #         return 0.0
            # r_safety_currLeader = _get_safety_reward(self.ego, self.ego.curr_leader)
            # if action_lateral == 2:
            #     r_safety_nextLeader = 0
            #     r_safety_nextFollower = 0
            # elif action_lateral == 1:
            #     r_safety_nextLeader = _get_safety_reward(self.ego, self.ego.trgt_leader)
            #     r_safety_nextFollower = _get_safety_reward(self.ego, self.ego.trgt_follower)
            # else:
            #     assert action_lateral == 0
            #     r_safety_nextLeader = _get_safety_reward(self.ego, self.ego.orig_leader)
            #     r_safety_nextFollower = _get_safety_reward(self.ego, self.ego.orig_follower)
            # r_safety = w_s * min(r_safety_currLeader, r_safety_nextLeader, r_safety_nextFollower)
            r_safety = 0
            r_total = r_comf + r_effi + r_safety
        else:
            # collision occurs
            r_comf = 0
            r_effi = 0
            r_safety = -300
            r_total = r_comf + r_effi + r_safety
        reward_dict = {'r_comf': r_comf, 'r_effi': r_effi, 'r_safety': r_safety}
        return r_total, reward_dict

    def updateReward3(self):
        return -self.ego.dis2tgtLane

    def is_done(self):
        # lane change successfully executed, episode ends, reset env
        # todo modify
        if self.is_success and self.success_timer*self.dt > 1:
            self.done = True
            # print('reset on: successfully lane change, dis2targetlane:',
            #       self.ego.dis2tgtLane)
        # too close to ramp entrance
        if self.ego.dis2entrance < 10.0:
            self.done = True
            # print('reset on: too close to ramp entrance, dis2targetlane:',
            #       self.ego.dis2tgtLane)
        # # ego vehicle out of env
        # if self.egoID not in self.vehID_tuple_all:
        #     self.done = True
        #     #print('reset on: self.ego not in env:', self.egoID not in self.vehID_tuple_all)
        # collision occurs
        self.collision_num = traci.simulation.getCollidingVehiclesNumber()
        if self.collision_num > 0:
            self.done = True
            self.is_done_info = 1
            # print('reset on: self.collision_num:', self.collision_num)

    def preStep(self):
        traci.simulationStep()
        self.vehID_tuple_all = traci.edge.getLastStepVehicleIDs(self.rd.entranceEdgeID)
        self.update_veh_dict(self.vehID_tuple_all)

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, call `reset()` outside env!! to reset this
        environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object):  longitudinal: action[0] = 1: accelerate
                                            action[0] = 2: decelerate
                                            action[0] = 0: acceleration = 0.0



                             **important**: orginal/target lane leader will not change despite the lateral position of
                                            the ego may change

                             lateral: action[1] = 1: lane change
                                      action[1] = 0: abort lane change, change back to original lane
                                      action[1] = 2: keep in current lateral position

        Returns:
            described in __init__
        """
        if action in [0, 1, 2]:
            action_lateral = 2
            action_longi = action
        else:
            assert action in [3, 4, 5]
            action_longi = 3
            if action == 3:
                action_lateral = 1
            elif action == 4:
                action_lateral = 2
            else:
                action_lateral = 0

        assert self.done is False, 'self.done is not False'
        assert action is not None, 'action is None'
        assert self.egoID in self.vehID_tuple_all, 'vehicle not in env'

        self.timestep += 1

        # lateral control-------------------------
        # episode in progress; 0:change back to original line; 1:lane change to target lane; 2:keep current
        # lane change to target lane

        if action_lateral == 1:
            self.is_success = self.ego.changeLane(True, self.ego.trgt_laneIndex, self.rd)
        # abort lane change, change back to ego's original lane
        if action_lateral == 0:
            self.is_success = self.ego.changeLane(True, self.ego.orig_laneIndex, self.rd)
        # keep current lateral position
        if action_lateral == 2:
            self.is_success = self.ego.changeLane(True, -1, self.rd)

        if self.is_success:
            self.success_timer += 1

        # longitudinal control2---------------------
        # clip minimum deceleration
        acceNext = max(self.ego.get_acceNext(action_longi), -4.5)
        vNext = max(self.ego.speed + acceNext * 0.1, 0.1)
        traci.vehicle.setSpeed(self.egoID, vNext)

        # update info------------------------------
        traci.simulationStep()
        self.vehID_tuple_all = traci.edge.getLastStepVehicleIDs(self.rd.entranceEdgeID)
        self.update_veh_dict(self.vehID_tuple_all)
        # check if episode ends
        self.is_done()
        if self.done is True:
            self.info['resetFlag'] = True
            self.reward, reward_dict = self.updateReward(action_lateral)
            self.info['reward_dict'] = reward_dict
            return self.observation, self.reward, self.done, self.info
        else:
            self.updateObservation()
            self.reward, reward_dict = self.updateReward(action_lateral)
            self.info['reward_dict'] = reward_dict
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
                for id in traci.edge.getLastStepVehicleIDs(self.rd.warmupEdge):
                    traci.vehicle.setLaneChangeMode(id, 0)
                if self.timestep > 1000:
                    raise Exception('cannot find ego after 1000 timesteps')

            assert self.egoID in self.vehID_tuple_all, "cannot start training while ego is not in env"
            self.done = False

            self.ego = self.veh_dict[self.egoID]
            self.ego.setTrgtLane(tlane)
            self.ego.update_info(self.rd, self.veh_dict)

            self.updateObservation()
            return self.observation
        return

    def close(self):
        traci.close()
