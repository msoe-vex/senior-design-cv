import numpy as np

class Quaternion():
    def __init__(self, qx: float, qy: float, qz: float, qw: float):
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw
    
    def __str__(self):
        return "Quaternion: [{0}, {1}, {2}, {3}]".format(round(self.qx, 3), round(self.qy, 3), round(self.qz, 3), round(self.qw, 3))


def getQuaternionFromEuler(roll: float, pitch: float, yaw: float) -> Quaternion:
    """
    Convert an Euler angle to a quaternion.

    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
    return Quaternion(qx, qy, qz, qw)

def rotDiff(q1: Quaternion,q2: Quaternion) -> Quaternion:
    """Finds the quaternion that, when applied to q1, will rotate an element to q2"""
    conjugate = Quaternion(q2.qx*-1,q2.qy*-1,q2.qz*-1,q2.qw)
    return rotAdd(q1,conjugate)

def rotAdd(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Finds the quaternion that is the equivalent to the rotation caused by both input quaternions applied sequentially."""
    w1 = q1.qw
    w2 = q2.qw
    x1 = q1.qx
    x2 = q2.qx
    y1 = q1.qy
    y2 = q2.qy
    z1 = q1.qz
    z2 = q2.qz

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return Quaternion(x,y,z,w)

def Quat2Mat(q: Quaternion) -> np.array:
    """Converts a quaternion to a rotation matrix (Numpy Array)"""
    m00 = 1 - 2 * q.qy**2 - 2 * q.qz**2
    m01 = 2 * q.qx * q.qy - 2 * q.qz * q.qw
    m02 = 2 * q.qx * q.qz + 2 * q.qy * q.qw
    m10 = 2 * q.qx * q.qy + 2 * q.qz * q.qw
    m11 = 1 - 2 * q.qx**2 - 2 * q.qz**2
    m12 = 2 * q.qy * q.qz - 2 * q.qx * q.qw
    m20 = 2 * q.qx * q.qz - 2 * q.qy * q.qw
    m21 = 2 * q.qy * q.qz + 2 * q.qx * q.qw
    m22 = 1 - 2 * q.qx**2 - 2 * q.qy**2
    result = [[m00,m01,m02],[m10,m11,m12],[m20,m21,m22]]

    return np.array(result)

def coordTransform(M: np.array, A: np.array, translation: np.array) -> np.array:
    """
    :param M: rotation matrix that represents the rotation between frames (q1 -> q2)
    :param A: vector of interest in frame q1
    :param translation: distance vector between origins of q1 and q2
    :return: vector A in relation to frame q2
    """
    APrime = np.matmul(M, A)
    q2Vec = APrime + translation
    return q2Vec


# Testing main
def main():
    camera = getQuaternionFromEuler(0,0.2618,0) #TODO - update for actual rotational changes in camera frame
    robot = Quaternion(0.0,0.0,0.0,1.0)
    cameraToRobotTranslation = np.array([0.19,0,0.3]) #TODO - update for actual translational changes in camera frame (X,Y,Z)

    rotatedMatrix = Quat2Mat(rotDiff(camera, robot))
    newVector = coordTransform(rotatedMatrix, np.array([1,1,2]), cameraToRobotTranslation) #TODO - just a test vector (actual comes from Joe's code)
    print(newVector)


if __name__ == '__main__':
    main()
