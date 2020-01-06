import sys
import os
import random
from env.LaneChangeEnv import LaneChangeEnv

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


def normal(env):
    egoid = 'lane1.2'
    ss = 6
    env.reset(egoid=egoid, tfc=2, sumoseed=ss, randomseed=3)
    traci.vehicle.setColor(egoid, (255, 69, 0))
    speedLimit = env.ego.speedLimit
    TH = 1
    env.ego.idm_obj.setSpeedLimit(speedLimit)
    env.ego.idm_obj.time_headway = TH
    # f = open('../data/idm_para_speedLimit{}_T{}.csv'.format(speedLimit, TH), 'a')
    # f.write('egoid, lanePos, dis2leader, speed, acce, leader_lanePos, leader_speed\n')

    for i in range(10000):
        # ss += 1

        obs, rwd, done, info = env.step(action=0)
        if done is True and info['resetFlag'] == 1:
            env.close()

        # f.write('%s, %s, %s, %s, %s, %s, %s\n' % (egoid, env.ego.pos_longi,
        #                                   env.ego.curr_leader.pos_longi - env.ego.pos_longi,
        #                                   env.ego.speed, env.ego.acce,
        #                                   env.ego.curr_leader.pos_longi, env.ego.curr_leader.speed))
        # f.flush()

def changeLane(env):
    egoid = 'lane1.2'
    ss = 6
    env.reset(egoid=egoid, tfc=0, sumoseed=ss, randomseed=3)
    traci.vehicle.setColor(egoid, (255, 69, 0))
    for i in range(10000):
        # ss += 1
        print(env.ego.pos_lat)
        obs, rwd, done, info = env.step(action=0*3+1)
        if done is True and info['resetFlag'] == 1:
            env.close()


def laneChange(low, high, origLane, tgtLane, rd):
    def changeNback(veh, cps):
        if veh.changeTimes < 25:
            # traci.vehicle.changeLane(veh.veh_id, veh.targetLane, 1)
            veh.changeLane(cps, veh.targetLane, rd)
            traci.vehicle.setColor(veh.veh_id, (255, 69, 0))
        elif 25 <= veh.changeTimes < 55:
            # traci.vehicle.changeLane(veh.veh_id, veh.origLane, 1)
            veh.changeLane(cps, veh.origLane, rd)
        else:
            traci.vehicle.changeSublane(veh.veh_id, 0.0)
        veh.changeTimes += 1



    #     elif veh.changeTimes == 0:
    #         veh.changeLane(cps, veh.targetLane, rd)
    #         veh.changeTimes += 1
    #
    # for vehID in env.veh_dict.keys():
    #     if int(vehID.split('.')[1]) % 2 == 0 and vehID.split('.')[0] == 'lane'+str(origLane):
    #     #if vehID == 'lane1.0':
    #         veh = env.veh_dict[vehID]
    #         if veh.lcPos is None:
    #             veh.lcPos = random.uniform(low, high)
    #         veh.targetLane = tgtLane
    #
    #         if veh.dis2entrance < veh.lcPos and abs(veh.pos_lat - (0.5+veh.targetLane)*rd.laneWidth) > 0.01:
    #
    #             if abs(veh.dis2entrance) > 20:
    #                 changeNback(veh, True)
    #
    #             else:
    #                 if not veh.laneIndex == 0:
    #                     traci.vehicle.setRouteID(veh.veh_id, rd.highwayKeepRouteID)
    #             # todo order of cmd, step, write

'''    
            if (abs(veh.pos_lat - (0.5+tgtLane)*3.2) < 0.01 or veh.dis2entrance < 1.0) \
               and veh.change_times == 1:

                #fe.write('%s, %s\n' % (vehID, veh.dis2entrance))
                fe.flush()
                veh.change_times += 1
'''


def extractAction(testid, env, f):
    if testid in env.vehID_tuple_all:
        veh = env.veh_dict[testid]
        # todo complete action extraction
        if veh.latAcce < 0 or veh.latSpeed < 0 and abs(veh.latAcce) < 0.1:
            action = 1
        else:
            action = 0

        f.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' % (
        veh.veh_id, veh.lanePos, veh.pos_lat, veh.speed, veh.latSpeed,
        bin(traci.vehicle.getLaneChangeState(veh.veh_id, -1)[1]),
        veh.latAcce, action, veh.laneIndex, veh.targetLane, veh.origLane))
        f.flush()


def IDMCtrl(env):
    f = open('data/IDM_mimic_original.csv', 'a')
    f.write('egoid, lanePos, dis2leader, speed, acce\n')

    egoid = 'lane1.2'
    env.reset(egoid=egoid, tfc=2, sumoseed=4, randomseed=3)
    traci.vehicle.setColor(egoid, (255, 69, 0))

    for step in range(10000):
        env.IDMStep()
        f.write('%s, %s, %s, %s, %s\n' % (egoid, env.veh_dict[egoid].lanePos,
                                          env.veh_dict[env.veh_dict[egoid].leaderID].lanePos - env.veh_dict[egoid].lanePos,
                                          env.veh_dict[egoid].speed, traci.vehicle.getAcceleration(egoid)))
        f.flush()


def badLongiCtrl(env):
    f = open('data/lateralCtr.csv', 'a')
    f.write('egoid, lanePos, dis2leader, speed, acce\n')

    egoid = 'lane1.2'
    env.reset(egoid=egoid, tfc=2, sumoseed=4, randomseed=3)
    traci.vehicle.setColor(egoid, (255, 69, 0))

    for step in range(10000):
        action = env.decision()
        obs, rwd, done, info = env.step(action)

        if done and info['resetFlag']:
            env.reset(egoid)

        #f.write('%s, %s, %s, %s, %s\n' % (egoid, obs[0][0], obs[1][0]-obs[0][0], obs[0][1], traci.vehicle.getAcceleration(egoid)))
        f.flush()


def doubleCtrl(env):
    f = open('data/doubleCtrl.csv', 'a')
    f.write('egoid, lanePos, lateralPos, speed, acce\n')

    egoid = 'lane1.1'
    env.reset(egoid=egoid, tfc=1, sumoseed=1, randomseed=3)
    traci.vehicle.setColor(egoid, (255, 69, 0))

    for i in range(10000):
        print(env.is_success)
        if env.ego.targetFollowerID != 'lane0.6' and env.ego.laneIndex == 1:
            obs, rwd, done, info = env.step(action=(1, 2))
        elif 2 > env.ego.lanePos - traci.vehicle.getLanePosition(env.ego.targetFollowerID) > -15:
            obs, rwd, done, info = env.step(action=(-1, 2))
        elif env.ego.lanePos - traci.vehicle.getLanePosition(env.ego.targetFollowerID) > 2 and not env.is_success:
            env.ego.targetLane = 0
            obs, rwd, done, info = env.step(action=(-1, 1))
        else:
            obs, rwd, done, info = env.step(action=(0, 2))

        if done is True and info['resetFlag'] == 1:
            env.reset(egoid=egoid, tfc=1, sumoseed=4, randomseed=3)
            traci.vehicle.setColor(egoid, (255, 69, 0))

        f.write('%s, %s, %s, %s, %s\n' % (egoid, obs[0][0], obs[0][2], obs[0][1], traci.vehicle.getAcceleration(egoid)))
        f.flush()


if __name__ == '__main__':

    #f = open('data/data21.csv', 'a')
    #f.write('egoid, lanePos, latPos, speed, latSpeed, lcState, latAcce, action, laneIndex, tgtlane, origLane\n')

    env = LaneChangeEnv()

    #badLongiCtrl(env)
    #IDMCtrl(env)
    normal(env)
    #changeLane(env)
    #doubleCtrl(env)
