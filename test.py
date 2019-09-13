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


def normalSim():
    # randomly make lane change or abort/keep
    if env.timestep < 130:
        obs, rwd, done, info = env.step(2)
    else:
        obs, rwd, done, info = env.step(1)

    print(env.timestep)
    if done is True and info['resetFlag'] == 1:
        env.reset(egoid='lane2.1', tlane=0, tfc=1, is_gui=True)


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


if __name__ == '__main__':

    f = open('data/data21.csv', 'a')
    f.write('egoid, lanePos, latPos, speed, latSpeed, lcState, latAcce, action, laneIndex, tgtlane, origLane\n')

    env = lcEnv.LaneChangeEnv()
    # env.reset(egoid='lane2.1', tlane=0, tfc=1, is_gui=True)
    #env.reset(None, tfc=1, sumoseed=3, randomseed=3)
    env.reset(egoid=None, tfc=1, sumoseed=3, randomseed=3)

    for step in range(10000):
        # todo emergency braking
        # todo use sumo computed vNext to perform lateral control, use moveToXY to perform lateral control
        laneChange(300, 350, 1, 0, env.rd)
        env.preStep()
        extractAction('lane1.0', env, f)

    #for step in range(10000):
        #env.demoStep()
