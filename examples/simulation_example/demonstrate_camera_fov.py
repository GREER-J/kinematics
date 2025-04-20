import numpy as np
from matplotlib import pyplot as plt
from kinematics_library import homogeneous as htf
from kinematics_library.homogeneous_utils import get_r_from_A
from src.camera_sensor import CameraSensor

cam = CameraSensor(1, 0, 0, 0)

pitch_ang = np.linspace(-cam.y_FOV, cam.y_FOV)
yaw_angle = np.linspace(-cam.z_FOV, cam.z_FOV)

vis_E = []
vis_N = []

invis_E = []
invis_N = []

for z in yaw_angle:
    for y in pitch_ang:
        A = cam.Apn@htf.roty(np.deg2rad(z)) @ htf.rotz(np.deg2rad(y)) @ htf.transx(10)
        r = get_r_from_A(A)
        res = cam.is_point_in_vis(r)
        r_N = float(r[1,0])
        r_E = float(r[0,0])
        if res:
            vis_E.append(r_N)
            vis_N.append(r_E)
        else:
            invis_E.append(r_N)
            invis_N.append(r_E)


print(len(vis_E))
print(len(vis_N))

print(vis_E[0])
print(vis_N[0])

print(vis_E[0:10])
print(vis_N[0:10])

print("Rounds complete")


plt.figure()
plt.scatter(vis_E, vis_N, 'go', label='visible points')
plt.scatter(invis_E, invis_N, 'ro', label='invisible points')

plt.show()
