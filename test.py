import gym
import sys
import os
import random
import LaneChangeEnv as lcEnv

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


def normal(env):
    f = open('data/original.csv', 'a')
    f.write('egoid, lanePos, dis2leader, speed, acce\n')

    egoid = 'lane1.2'
    env.reset(egoid=egoid, tfc=2, sumoseed=4, randomseed=3)
    traci.vehicle.setColor(egoid, (255, 69, 0))

    for i in range(10000):
        obs, rwd, done, info = env.step()
        if done is True and info['resetFlag'] == 1:
            env.reset(egoid=egoid, tfc=2, sumoseed=4, randomseed=3)
            traci.vehicle.setColor(egoid, (255, 69, 0))

        f.write('%s, %s, %s, %s, %s\n' % (egoid, env.veh_dict[egoid].lanePos,
                                          env.veh_dict[env.veh_dict[egoid].leaderID].lanePos - env.veh_dict[
                                              egoid].lanePos,
                                          env.veh_dict[egoid].speed, traci.vehicle.getAcceleration(egoid)))
        f.flush()


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

    def changeLane(mutiple, veh, cps):
        traci.vehicle.setColor(veh.veh_id, (255, 69, 0))
        if mutiple is True:
            veh.changeLane(cps, veh.targetLane, rd)
            # traci.vehicle.moveToXY('entranceEdge_1')
            # traci.vehicle.changeLane(veh.veh_id, veh.targetLane, 1)
            # todo: set route affects lc behavior, changeSublane doesn't work, changeLane works
            # traci.vehicle.setRouteID(veh.veh_id, rd.rampExitRouteID)

        elif veh.changeTimes == 0:
            veh.changeLane(cps, veh.targetLane, rd)
            veh.changeTimes += 1

    for vehID in env.veh_dict.keys():
        if int(vehID.split('.')[1]) % 2 == 0 and vehID.split('.')[0] == 'lane'+str(origLane):
        #if vehID == 'lane1.0':
            veh = env.veh_dict[vehID]
            if veh.lcPos is None:
                veh.lcPos = random.uniform(low, high)
            veh.targetLane = tgtLane

            if veh.dis2entrance < veh.lcPos and abs(veh.pos_lat - (0.5+veh.targetLane)*rd.laneWidth) > 0.01:

                if abs(veh.dis2entrance) > 20:
                    changeNback(veh, True)

                else:
                    if not veh.laneIndex == 0:
                        traci.vehicle.setRouteID(veh.veh_id, rd.highwayKeepRouteID)
                # todo order of cmd, step, write

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

    env = lcEnv.LaneChangeEnv()

    #badLongiCtrl(env)
    #IDMCtrl(env)
    normal(env)
    #doubleCtrl(env)
