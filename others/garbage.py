def calcAcce(self):
    if self.leaderID is None and self.targetLeaderID is None:
        print('no leader')
        acceNext = self.idm_obj.calc_acce(self.speed, None, None)
    elif self.leaderID is None and self.targetLeaderID is not None:
        print('following target leader')
        acceNext = self.idm_obj.calc_acce(self.speed, traci.vehicle.getLanePosition(self.targetLeaderID) - self.lanePos,
                                          traci.vehicle.getSpeed(self.targetLeaderID))
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



    # longitudinal control1---------------------
    '''choice 1 of longitudinal actions
    if action_longi != 0:
        self.longiCtrl(action_longi)

    if action_longi == 0:
        # follow leader in current lane
        acceNext = self.ego.updateLongitudinalSpeedIDM()
    else:
        print('action_longi', action_longi)
        assert action_longi == 1, 'action_longi invalid!'
        acceNext = self.ego.calcAcce()
    '''

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


'''
        if self.is_ego == 1:
            self.dis2tgtLane = abs((0.5 + self.trgt_laneIndex) * rd.laneWidth - self.pos_lat)
            self.dis2entrance = rd.laneLength - self.lanePos

            # get current leader information
            leader_tuple = traci.vehicle.getLeader(self.veh_id)
            if leader_tuple is not None:
                if leader_tuple[0] in list(veh_dict.keys()):
                    self.curr_leader['id'] = leader_tuple[0]
                    self.curr_leader['dis'] = leader_tuple[1]
                    self.curr_leader['speed'] = traci.vehicle.getSpeed(leader_tuple[0])
                else:
                    self.curr_leader = {'id': None, 'dis': None, 'speed': None}
            else:
                self.curr_leader = {'id': None, 'dis': None, 'speed': None}

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
                    self.orig_follower['id'] = follower_id
                    self.orig_follower['dis'] = -min_dis
                    self.orig_follower['speed'] = traci.vehicle.getSpeed(follower_id)
                else:
                    self.orig_follower = {'id': None, 'dis': None, 'speed': None}
                # target lane leader
                leaders_list = traci.vehicle.getNeighbors(self.veh_id, 1+2+0)
                if len(leaders_list) != 0:
                    if leaders_list[0][0] in list(veh_dict.keys()):
                        self.trgt_leader['id'] = leaders_list[0][0]
                        self.trgt_leader['dis'] = leaders_list[0][1]
                        self.trgt_leader['speed'] = traci.vehicle.getSpeed(leaders_list[0][0])
                    else:
                        self.trgt_leader = {'id': None, 'dis': None, 'speed': None}
                else:
                    self.trgt_leader = {'id': None, 'dis': None, 'speed': None}
                # target lane follower
                followers_list = traci.vehicle.getNeighbors(self.veh_id, 1+0+0)
                if len(followers_list) != 0:
                    if followers_list[0][0] in list(veh_dict.keys()):
                        self.trgt_follower['id'] = followers_list[0][0]
                        self.trgt_follower['dis'] = followers_list[0][1]
                        self.trgt_follower['speed'] = traci.vehicle.getSpeed(followers_list[0][0])
                    else:
                        self.trgt_follower = {'id': None, 'dis': None, 'speed': None}
                else:
                    self.trgt_follower = {'id': None, 'dis': None, 'speed': None}

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
                    self.trgt_follower['id'] = follower_id
                    self.trgt_follower['dis'] = -min_dis
                    self.trgt_follower['speed'] = traci.vehicle.getSpeed(follower_id)
                else:
                    self.trgt_follower = {'id': None, 'dis': None, 'speed': None}
                # original lane leader
                leaders_list = traci.vehicle.getNeighbors(self.veh_id, 0 + 2 + 0)
                if len(leaders_list) != 0:
                    if leaders_list[0][0] in list(veh_dict.keys()):
                        self.orig_leader['id'] = leaders_list[0][0]
                        self.orig_leader['dis'] = leaders_list[0][1]
                        self.orig_leader['speed'] = traci.vehicle.getSpeed(leaders_list[0][0])
                    else:
                        self.orig_leader = {'id': None, 'dis': None, 'speed': None}
                else:
                    self.orig_leader = {'id': None, 'dis': None, 'speed': None}
                # original lane follower
                followers_list = traci.vehicle.getNeighbors(self.veh_id, 0 + 0 + 0)
                if len(followers_list) != 0:
                    if followers_list[0][0] in list(veh_dict.keys()):
                        self.orig_follower['id'] = followers_list[0][0]
                        self.orig_follower['dis'] = followers_list[0][1]
                        self.orig_follower['speed'] = traci.vehicle.getSpeed(followers_list[0][0])
                    else:
                        self.orig_follower = {'id': None, 'dis': None, 'speed': None}
                else:
                    self.orig_follower = {'id': None, 'dis': None, 'speed': None}
                    '''

'''
        summary_eval_multi_steps = {}
        summary_actor_loss_multi_steps = []
        summary_critic_loss_multi_steps = []
        # update actor
        # clipping method, find this is better (OpenAI's paper)
        for _ in range(A_UPDATE_STEPS):
            self.actor_step += 1
            to, smr_temp = self.sess.run([self.atrain_op, self.summary_dict['actor_loss']],
                                         feed_dict={self.tfs: s, self.tfa: a, self.tfadv: adv})
            summary_actor_loss_multi_steps.append(smr_temp)
        summary_eval_multi_steps['actor_loss'] = summary_actor_loss_multi_steps
        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.critic_step += 1
            to, smr_temp = self.sess.run([self.ctrain_op, self.summary_dict['critic_loss']],
                                         feed_dict={self.tfs: s, self.tfdc_r: r})
            summary_critic_loss_multi_steps.append(smr_temp)
        summary_eval_multi_steps['critic_loss'] = summary_critic_loss_multi_steps
        return summary_eval_multi_steps
        '''


def _get_reward_safety(ego, veh_temp):
    if veh_temp is not None:
        # compute relative distance
        vec_pos_ego = np.array([ego.pos_longi, ego.pos_lat])
        vec_pos_temp = np.array([veh_temp.pos_longi, veh_temp.pos_lat])
        delta_vec_pos = vec_pos_ego - vec_pos_temp
        delta_pos_abs = np.linalg.norm(delta_vec_pos, 2)
        assert delta_pos_abs >= 0
        if delta_pos_abs <= 20:
            # compute relative velocity
            vec_vel_ego = np.array([ego.speed, ego.speed_lat])
            vec_vel_temp = np.array([veh_temp.speed, veh_temp.speed_lat])
            delta_vec_vel = vec_vel_ego - vec_vel_temp
            inner_product = np.dot(delta_vec_pos, delta_vec_vel)
            if inner_product < 0:
                vel_projected = inner_product / np.linalg.norm(delta_vec_pos, 2)
                TTC = delta_pos_abs / -vel_projected
                assert TTC >= 0
                return -1 / (TTC + 0.01)
            else:
                return 0.0
        else:
            return 0.0
    else:
        return 0.0


r_safety_orig_leader = _get_reward_safety(self.ego, self.ego.orig_leader)
r_safety_orig_follower = _get_reward_safety(self.ego, self.ego.orig_follower)
r_safety_trgt_leader = _get_reward_safety(self.ego, self.ego.trgt_leader)
r_safety_trgt_follower = _get_reward_safety(self.ego, self.ego.trgt_follower)
r_safety = w_s * min(r_safety_orig_leader, r_safety_orig_follower, r_safety_trgt_leader, r_safety_trgt_follower)
