import numpy as np

class CatmullRomSpline:
    def __init__(self, dim, num_nodes, cyclic):
        self.dim = dim
        self.num_nodes = num_nodes
        self.cyclic = cyclic
        self.nodes = np.zeros((self.num_nodes, dim))

        self.interp_mat = np.array([[-1,3,-3,1], [2,-5,4,-1], [-1,0,1,0], [0,2,0,0]]) * 0.5

    def get_current_parameters(self):
        if self.cyclic:
            return np.reshape(self.nodes[:-1], (np.prod(self.nodes.shape) - self.dim,))
        else:
            return np.reshape(self.nodes, (np.prod(self.nodes.shape), ))

    def set_parameters(self, param):
        if self.cyclic:
            param = np.concatenate([param, param[0:self.dim]])
        self.nodes = np.reshape(param, self.nodes.shape)

    def get_interpolated_points(self, interp_num):
        """
        Get the interpolated point positions.

        :param interp_num: number of interpolated point in between control points
        :return: an array of interpolated points
        """

        control_points = []
        if not self.cyclic:
            control_points.append(self.nodes[0])
        else:
            control_points.append(self.nodes[-2])
        control_points += self.nodes.tolist()
        if not self.cyclic:
            control_points.append(self.nodes[-1])
            # control_points.append(self.nodes[-1])
        else:
            control_points.append(self.nodes[1])
            # control_points.append(self.nodes[2])

        interpolated_points = []
        for i in range(self.num_nodes-1):
            point_mat = np.array(control_points[i:i+4])
            interp_ts = np.arange(interp_num) / interp_num
            interp_t_mat = np.concatenate([[interp_ts ** 3], [interp_ts ** 2], [interp_ts ** 1],  [interp_ts ** 0]], axis=0).T

            interp_p = np.dot(np.dot(interp_t_mat, self.interp_mat), point_mat)

            interpolated_points += interp_p.tolist()

        interpolated_points = np.array(interpolated_points)

        return interpolated_points
