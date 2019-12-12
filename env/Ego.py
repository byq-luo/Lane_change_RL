import math
import os, sys
from env.Vehicle import Vehicle
from collections import deque
from env.IDM import IDM

# *******************************
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


class Ego(Vehicle):
    def __init__(self, veh_id, rd):
        super(Ego, self).__init__(veh_id, rd)
        self.trgt_laneIndex = self.curr_laneIndex
        self.orig_laneIndex = self.curr_laneIndex
        # surrounding vehicle information
        self.curr_leader = None
        self.orig_leader = None
        self.orig_follower = None
        self.trgt_leader = None
        self.trgt_follower = None
        # navigation information
        self.dis2tgtLane = None
        self.dis2entrance = None
        # idm object
        traci.vehicle.setSpeedMode(self.veh_id, 0)
        self.speedFactor = traci.vehicle.getSpeedFactor(self.veh_id)
        self.speedLimit = self.speedFactor * traci.lane.getMaxSpeed(traci.vehicle.getLaneID(self.veh_id))
        self.idm_obj = IDM(v0=self.speedLimit)

    def update_info(self, rd, veh_dict):
        super(Ego, self).update_info(rd, veh_dict)
        self.dis2tgtLane = abs((0.5 + self.trgt_laneIndex) * rd.laneWidth - self.pos_lat)
        self.dis2entrance = rd.laneLength - self.pos_longi

        # get current leader information
        leader_tuple = traci.vehicle.getLeader(self.veh_id)
        if leader_tuple is not None:
            if leader_tuple[0] in list(veh_dict.keys()):
                self.curr_leader = veh_dict[leader_tuple[0]]
            else:
                self.curr_leader = None
        else:
            self.curr_leader = None

        # the following 2 all in list [(id1, distance1), (id2, distance2), ...], each tuple is a leader on one lane
        # for a single left lane, the list only contains 1 tuple, eg.[(id1, distance1)]

        # get target lane leader&follower information
        if self.curr_laneIndex > self.trgt_laneIndex:
            assert self.curr_laneIndex == self.orig_laneIndex
            # original lane leader
            self.orig_leader = self.curr_leader
            # original lane follower
            follower_id = None
            min_dis = 100000
            for veh_id in traci.lane.getLastStepVehicleIDs(rd.entranceEdgeID+'_'+str(self.curr_laneIndex)):
                dis_temp = self.pos_longi - veh_dict[veh_id].pos_longi
                if dis_temp > 0:
                    if dis_temp < min_dis:
                        follower_id = veh_id
                        min_dis = dis_temp

            if follower_id is not None:
                self.orig_follower = veh_dict[follower_id]
            else:
                self.orig_follower = None
            # target lane leader
            leaders_list = traci.vehicle.getNeighbors(self.veh_id, 1+2+0)
            if len(leaders_list) != 0:
                if leaders_list[0][0] in list(veh_dict.keys()):
                    self.trgt_leader = veh_dict[leaders_list[0][0]]
                else:
                    self.trgt_leader = None
            else:
                self.trgt_leader = None
            # target lane follower
            followers_list = traci.vehicle.getNeighbors(self.veh_id, 1+0+0)
            if len(followers_list) != 0:
                if followers_list[0][0] in list(veh_dict.keys()):
                    self.trgt_follower = veh_dict[followers_list[0][0]]
                else:
                    self.trgt_follower = None
            else:
                self.trgt_follower = None

        else:
            assert self.curr_laneIndex == self.trgt_laneIndex
            # target lane leader
            self.trgt_leader = self.curr_leader
            # target lane follower
            follower_id = None
            min_dis = 100000
            for veh_id in traci.lane.getLastStepVehicleIDs(rd.entranceEdgeID + '_' + str(self.curr_laneIndex)):
                dis_temp = self.pos_longi - veh_dict[veh_id].pos_longi
                if dis_temp > 0:
                    if dis_temp < min_dis:
                        follower_id = veh_id
                        min_dis = dis_temp

            if follower_id is not None:
                self.trgt_follower = veh_dict[follower_id]
            else:
                self.trgt_follower = None
            # original lane leader
            leaders_list = traci.vehicle.getNeighbors(self.veh_id, 0 + 2 + 0)
            if len(leaders_list) != 0:
                if leaders_list[0][0] in list(veh_dict.keys()):
                    self.orig_leader = veh_dict[leaders_list[0][0]]
                else:
                    self.orig_leader = None
            else:
                self.orig_leader = None
            # original lane follower
            followers_list = traci.vehicle.getNeighbors(self.veh_id, 0 + 0 + 0)
            if len(followers_list) != 0:
                if followers_list[0][0] in list(veh_dict.keys()):
                    self.orig_follower = veh_dict[followers_list[0][0]]
                else:
                    self.orig_follower = None
            else:
                self.orig_follower = None

    def setTrgtLane(self, trgtlane):
        self.trgt_laneIndex = trgtlane

    def get_acceNext(self, action_longi):
        if action_longi == 0:
            return 1.5
        elif action_longi == 1:
            return -1.5
        elif action_longi == 2:
            return 0
        else:
            assert action_longi == 3
            return self.updateLongitudinalSpeedIDM(2)

    def updateLongitudinalSpeedIDM(self, action):
        """
        use IDM to control vehicle speed
        :return: acceleration for next timestep
        """
        # cannot acquire vNext, compute longitudinal speed on our own
        # determine leader
        if action == 0:
            leader = self.orig_leader
        elif action == 1:
            leader = self.trgt_leader
        else:
            assert action == 2
            leader = self.curr_leader
        # compute acceNext
        if leader is not None:
            leaderDis = leader.pos_longi - self.pos_longi
            acceNext = self.idm_obj.calc_acce(self.speed, leaderDis, leader.speed)
        else:
            acceNext = self.idm_obj.calc_acce(self.speed, None, None)
        return acceNext

    def changeLane(self, cps, tgtlane, rd):
        '''
        # make compulsory/default lane change, do not respect other vehicles
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
