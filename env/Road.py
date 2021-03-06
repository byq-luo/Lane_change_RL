import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


class Road:
    def __init__(self):
        """
        assume all lanes have the same width
        """
        self.warmupEdge = 'warm_up'
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