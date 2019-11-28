import matplotlib.pyplot as plt
import pandas as pd


def read_IDM_data():
    # egoid, lanePos, dis2leader, speed, acce
    df_speed_idm = pd.read_csv('../data/IDM_mimic_original.csv', usecols=[' lanePos', ' speed'])
    df_speed_orig = pd.read_csv('../data/original.csv', usecols=[' lanePos', ' speed'])
    speed_idm = df_speed_idm.to_numpy()
    speed_orig = df_speed_orig.to_numpy()

    plt.figure(figsize=(18, 6))
    plt.plot(speed_idm[:, 0], speed_idm[:, 1], label='Our IDM', color='b')
    plt.plot(speed_orig[:, 0], speed_orig[:, 1], label='SUMO IDM', color='r')

    plt.title('comparison of speed using different longitudinal control', fontsize=20)
    plt.xlabel('lanePos (m)', fontsize=16)
    plt.ylabel(r'speed $(m/s^2)$', fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)

    #plt.savefig('../figures/SUMO_vs_own_IDM')
    plt.show()


def IDM_para():
    data_1 = pd.read_csv('../data/idm_para_speedLimit40_T0.5.csv',
                         usecols=[' lanePos', ' dis2leader', ' speed']).to_numpy()
    data_2 = pd.read_csv('../data/idm_para_speedLimit40_T1.csv',
                         usecols=[' lanePos', ' dis2leader', ' speed']).to_numpy()
    data_3 = pd.read_csv('../data/idm_para_speedLimit40_T0.1.csv',
                         usecols=[' lanePos', ' dis2leader', ' speed']).to_numpy()
    plt.figure(figsize=(18, 10))
    plt.subplot(2, 1, 1)
    plt.plot(data_1[:, 0], data_1[:, 2], label='v0=40, T=0.5')
    plt.plot(data_2[:, 0], data_2[:, 2], label='v0=40, T=1')
    plt.plot(data_3[:, 0], data_3[:, 2], label='v0=40, T=0.1')
    plt.title('speed'), plt.grid(), plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(data_1[:, 0], data_1[:, 1], label='v0=40, T=0.5')
    plt.plot(data_2[:, 0], data_2[:, 1], label='v0=40, T=1')
    plt.plot(data_3[:, 0], data_3[:, 1], label='v0=40, T=0.1')
    plt.title('dis2leader'), plt.grid(), plt.legend()
    plt.savefig('../figures/IDM_different_paras')
    plt.show()
    #plt.savefig('../figures/IDM_different_paras')

def IDM_normal():
    data = pd.read_csv('../data/idm_para_speedLimit33.35816218271343_T1.csv',
                         usecols=[' lanePos', ' dis2leader', ' speed', ' leader_lanePos', ' leader_speed']).to_numpy()
    data = data[:125, :]
    plt.figure(figsize=(18, 11))
    plt.subplot(2, 1, 1)
    plt.plot(data[:, 0], data[:, 2], label='ego')
    plt.plot(data[:, 0], data[:, 4], label='leader')
    plt.title('longitudinal speed'), plt.xlabel('lane pos / m'), plt.ylabel('speed / m/s')
    plt.grid(), plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(data[:, 0], data[:, 1])
    # plt.plot(data[:, 2])
    plt.title('distance to leader'), plt.xlabel('ego lane pos / m'), plt.ylabel('distance to leader / m')
    plt.grid()
    plt.savefig('../figures/IDM_normal')
    plt.show()

if __name__ == '__main__':
    # read_IDM_data()
    # IDM_para()
    IDM_normal()
