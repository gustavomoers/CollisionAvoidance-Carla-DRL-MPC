import numpy as np
import pickle
import matplotlib.pyplot as plt

MPC_record = pickle.load( open( "MPC_record.pkl", "rb" ) )
PID_record = pickle.load( open( "PID_record.pkl", "rb" ) )

plt.figure()
plt.plot(MPC_record["time"], MPC_record["speed"], label='MPC Speed')
plt.plot(PID_record["time"], PID_record["speed"], label='PID Speed')
plt.legend(loc='best')
plt.ylabel('speed (m/s)')
plt.xlabel('simulation time (s)')
plt.title('Vehicle speed over time')
plt.xlim(0, 60)
plt.show()


plt.figure()
plt.plot(MPC_record["time"], MPC_record["yaw"], label='MPC yaw')
plt.plot(PID_record["time"], PID_record["yaw"], label='PID yaw')
plt.legend(loc='best')
plt.ylabel('degree')
plt.xlabel('simulation time (s)')
plt.title('Vehicle yaw over time')
plt.xlim(0, 60)
plt.show()
