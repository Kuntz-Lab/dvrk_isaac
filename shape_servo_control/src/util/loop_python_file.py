
import sys
import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
pkg_path = "/home/baothach/dvrk_ws"
# sys.path.append(pkg_path + '/src/test_stuff')
import os
i = 0
# os.chdir(pkg_path + '/src/test_stuff')
os.chdir(pkg_path)
while i < 1:
    # os.pause(2)
    # os.system("python3 nothing.py")
    os.system("source devel/setup.bash")
    # os.system("rosrun shape_servo_control nothing.py")
    os.system("rosrun shape_servo_control test_different_size_objects.py --flex")
    i += 1