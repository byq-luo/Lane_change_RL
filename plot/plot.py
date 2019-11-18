import matplotlib.pyplot as plt
import pandas as pd
f = open('data/original.csv', 'a')


def read_IDM_data():
    # egoid, lanePos, dis2leader, speed, acce
    df_speed_idm = pd.read_csv('data/IDM_mimic_original.csv', usecols=[' lanePos', ' speed'])
    df_speed_orig = pd.read_csv('data/original.csv', usecols=[' lanePos', ' speed'])
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

    plt.savefig('figures/SUMO_vs_own_IDM')
    plt.show()


if __name__ == '__main__':
    read_IDM_data()
