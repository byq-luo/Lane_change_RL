import math
import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


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

        self.curr_laneIndex = traci.vehicle.getLaneIndex(self.veh_id)
        self.trgt_laneIndex = self.curr_laneIndex
        self.orig_laneIndex = self.curr_laneIndex

        self.pos_lat = traci.vehicle.getLateralLanePosition(self.veh_id) + (self.curr_laneIndex+0.5)*rd.laneWidth
        self.pos_lat_last = self.pos_lat
        self.lanePos = traci.vehicle.getLanePosition(self.veh_id)
        self.yawAngle = 0

        # todo use Vehicle class directly
        # self.curr_leader = {'id': None, 'obj':None, 'dis':None, 'speed':None}
        # self.orig_leader = {'id': None, 'obj':None, 'dis':None, 'speed':None}
        # self.trgt_leader = {'id': None, 'obj':None, 'dis':None, 'speed':None}
        # self.trgt_follower = {'id': None, 'obj':None, 'dis':None, 'speed':None}

        # self.curr_leader = {'id': None, 'dis': None, 'speed': None}
        # self.orig_leader = {'id': None, 'dis': None, 'speed': None}
        # self.orig_follower = {'id': None, 'dis': None, 'speed': None}
        # self.trgt_leader = {'id': None, 'dis': None, 'speed': None}
        # self.trgt_follower = {'id': None, 'dis': None, 'speed': None}

        self.curr_leader = None
        self.orig_leader = None
        self.orig_follower = None
        self.trgt_leader = None
        self.trgt_follower = None

        self.dis2tgtLane = None
        self.dis2entrance = None
        self.lcPos = None
        self.reward = 0
        # not update every step

        self.is_ego = 0
        self.changeTimes = 0
        self.idm_obj = None

        traci.vehicle.setLaneChangeMode(veh_id, 256)  # 768

    def update_info(self, rd, veh_dict):
        self.curr_laneIndex = traci.vehicle.getLaneIndex(self.veh_id)
        self.speed = traci.vehicle.getSpeed(self.veh_id)

        self.acce = traci.vehicle.getAcceleration(self.veh_id)
        self.delta_acce = (self.acce - self.acce_last) / 0.1
        self.acce_last = self.acce

        self.pos_lat_last = self.pos_lat
        self.pos_lat = traci.vehicle.getLateralLanePosition(self.veh_id) + (self.curr_laneIndex+0.5)*rd.laneWidth
        self.latSpeed_last = self.latSpeed
        self.latSpeed = (self.pos_lat - self.pos_lat_last) / 0.1  # 0.1 for time step length
        self.latAcce = (self.latSpeed - self.latSpeed_last) / 0.1

        self.pos = traci.vehicle.getPosition(self.veh_id)
        self.pos_x = self.pos[0]
        self.pos_y = self.pos[1]
        self.lanePos = traci.vehicle.getLanePosition(self.veh_id)
        self.yawAngle = math.atan(self.latSpeed / max(self.speed, 0.00000001))

        if self.is_ego == 1:
            self.dis2tgtLane = abs((0.5 + self.trgt_laneIndex) * rd.laneWidth - self.pos_lat)
            self.dis2entrance = rd.laneLength - self.lanePos

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
                    dis_temp = self.lanePos - veh_dict[veh_id].lanePos
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
                    dis_temp = self.lanePos - veh_dict[veh_id].lanePos
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

    def updateLongitudinalSpeedIDM(self, action):
        """
        use IDM to control vehicle speed
        :return:
        """
        # cannot acquire vNext, compute longitudinal speed on our own
        # determine leader
        if action == 0:
            leader = self.orig_leader
        else:
            assert action == 1
            leader = self.trgt_leader
        # compute acceNext
        if leader is not None:
            leaderDis = leader.lanePos - self.lanePos
            acceNext = self.idm_obj.calc_acce(self.speed, leaderDis, leader.speed)
        else:
            acceNext = self.idm_obj.calc_acce(self.speed, None, None)

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


class Ego:
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

        self.curr_laneIndex = traci.vehicle.getLaneIndex(self.veh_id)
        self.trgt_laneIndex = self.curr_laneIndex
        self.orig_laneIndex = self.curr_laneIndex

        self.pos_lat = traci.vehicle.getLateralLanePosition(self.veh_id) + (self.curr_laneIndex+0.5)*rd.laneWidth
        self.pos_lat_last = self.pos_lat
        self.lanePos = traci.vehicle.getLanePosition(self.veh_id)
        self.yawAngle = 0

        # todo use Vehicle class directly
        # self.curr_leader = {'id': None, 'obj':None, 'dis':None, 'speed':None}
        # self.orig_leader = {'id': None, 'obj':None, 'dis':None, 'speed':None}
        # self.trgt_leader = {'id': None, 'obj':None, 'dis':None, 'speed':None}
        # self.trgt_follower = {'id': None, 'obj':None, 'dis':None, 'speed':None}

        # self.curr_leader = {'id': None, 'dis': None, 'speed': None}
        # self.orig_leader = {'id': None, 'dis': None, 'speed': None}
        # self.orig_follower = {'id': None, 'dis': None, 'speed': None}
        # self.trgt_leader = {'id': None, 'dis': None, 'speed': None}
        # self.trgt_follower = {'id': None, 'dis': None, 'speed': None}

        self.curr_leader = None
        self.orig_leader = None
        self.orig_follower = None
        self.trgt_leader = None
        self.trgt_follower = None

        self.dis2tgtLane = None
        self.dis2entrance = None
        self.lcPos = None
        self.reward = 0
        # not update every step

        self.is_ego = 0
        self.changeTimes = 0
        self.idm_obj = None

        traci.vehicle.setLaneChangeMode(veh_id, 256)  # 768

    def update_info(self, rd, veh_dict):
        self.curr_laneIndex = traci.vehicle.getLaneIndex(self.veh_id)
        self.speed = traci.vehicle.getSpeed(self.veh_id)

        self.acce = traci.vehicle.getAcceleration(self.veh_id)
        self.delta_acce = (self.acce - self.acce_last) / 0.1
        self.acce_last = self.acce

        self.pos_lat_last = self.pos_lat
        self.pos_lat = traci.vehicle.getLateralLanePosition(self.veh_id) + (self.curr_laneIndex+0.5)*rd.laneWidth
        self.latSpeed_last = self.latSpeed
        self.latSpeed = (self.pos_lat - self.pos_lat_last) / 0.1  # 0.1 for time step length
        self.latAcce = (self.latSpeed - self.latSpeed_last) / 0.1

        self.pos = traci.vehicle.getPosition(self.veh_id)
        self.pos_x = self.pos[0]
        self.pos_y = self.pos[1]
        self.lanePos = traci.vehicle.getLanePosition(self.veh_id)
        self.yawAngle = math.atan(self.latSpeed / max(self.speed, 0.00000001))

        if self.is_ego == 1:
            self.dis2tgtLane = abs((0.5 + self.trgt_laneIndex) * rd.laneWidth - self.pos_lat)
            self.dis2entrance = rd.laneLength - self.lanePos

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
                    dis_temp = self.lanePos - veh_dict[veh_id].lanePos
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
                    dis_temp = self.lanePos - veh_dict[veh_id].lanePos
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

    def updateLongitudinalSpeedIDM(self, action):
        """
        use IDM to control vehicle speed
        :return:
        """
        # cannot acquire vNext, compute longitudinal speed on our own
        # determine leader
        if action == 0:
            leader = self.orig_leader
        else:
            assert action == 1
            leader = self.trgt_leader
        # compute acceNext
        if leader is not None:
            leaderDis = leader.lanePos - self.lanePos
            acceNext = self.idm_obj.calc_acce(self.speed, leaderDis, leader.speed)
        else:
            acceNext = self.idm_obj.calc_acce(self.speed, None, None)

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