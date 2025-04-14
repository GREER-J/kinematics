import numpy as np

"""
MATLAB implementation
function S = skew(u)
%skew    Obtain matrix operator for left action of cross product
%   S = skew(u)
%   takes column vector u of length 3 and returns a matrix S,
%   such that
%       S*v = cross(u,v)
%
%   See also cross.

S = [ ...
    0,-u(3),u(2); ...
    u(3),0,-u(1); ...
    -u(2),u(1),0 ...
    ];
"""


def skew(vec: np.ndarray) -> np.ndarray:
    a1 = vec[0, 0]
    a2 = vec[1, 0]
    a3 = vec[2, 0]
    return np.matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])
