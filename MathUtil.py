import numbers
from math import sqrt, pi, acos, sin, cos, e
from random import random

EPSILON = 0.00001


def clip(val, lower=0.0, upper=1.0):
    """
    Clips val between lower and upper.

    >>> clip(1, 0, 2)
    1
    >>> clip(2, 3, 6)
    3
    >>> clip(5, 1, 2)
    2

    Works recursively on lists.

    >>> clip([-0.2, 0.5, 1.4, 0.7])
    [0.0, 0.5, 1.0, 0.7]

    :param val: value to be clipped
    :param lower: lower bound
    :param upper: upper bound
    :return: val clipped between lower and upper
    """
    if isinstance(val, list):
        return [clip(v, lower, upper) for v in val]
    return max(lower, min(upper, val))


class Vec2:
    """
    Vector class with an X and Y component.
    """

    def __init__(self, xx=None, yy=None):
        """
        Initialize Vec2 object.

        If xx or yy is not filled in, they will be initialized as follows:

        >>> Vec2() == Vec2(0, 0)
        True
        >>> Vec2(1) == Vec2(1, 1)
        True
        >>> Vec2(2, 3) == Vec2(2, 3)
        True

        :param xx: X coordinate of the vector.
        :param yy: Y coordinate of the vector.
        """
        if xx is None:
            self.x = 0
            self.y = 0

        elif isinstance(xx, Vec2):
            self.x = xx.x
            self.y = xx.y

        elif yy is None:
            self.x = xx
            self.y = xx
        else:
            self.x = xx
            self.y = yy

    def normalize(self):
        """
        Normalize self.

        This will set the length of self to 1.

        >>> v = Vec2(2, 0)
        >>> v.normalize()
        >>> v == Vec2(1, 0)
        True

        :return: None
        """
        length = self.length()
        if length > 0:
            self.x /= length
            self.y /= length

    def unit(self):
        """
        Return normalized version of self.

        >>> v = Vec2(0, 4)
        >>> v.unit() == Vec2(0, 1)
        True

        :return: normalized version of self.
        """
        length = self.length()
        if length > 0:
            return Vec2(self.x / length, self.y / length)
        return self

    def __mul__(self, other):
        """
        Element-wise Multiply self * other.

        Can work for other Vec2 or scalar

        >>> Vec2(2,4) * Vec2(3, 2) == Vec2(6, 8)
        True
        >>> Vec2(5,3) * 2 == Vec2(10, 6)
        True

        :param other: multiplicand
        :return: self * other
        """
        if isinstance(other, numbers.Number):
            return Vec2(self.x * other, self.y * other)
        elif isinstance(other, Vec2):
            return Vec2(self.x * other.x, self.y * other.y)

    def dot(self, other):
        """
        Return dot product of self and other.

        >>> Vec2(3, 4).dot(Vec2(2, 3))
        18

        :param other: multiplicand
        :return: self dot other
        """
        return self.x * other.x + self.y * other.y

    @staticmethod
    def random_uv():
        a = 1
        b = 1

        while a + b > 1:
            a = random()
            b = random()

        return Vec2(a, b)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def length2(self):
        return self.x ** 2 + self.y ** 2

    def length(self):
        return sqrt(self.length2())

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        return "Vec2<{} {}>".format(self.x, self.y)

        # no support for this on the HPC 10 cluster
        # return f"Vec2<{self.x} {self.y}>"

    def __eq__(self, other):
        if abs(self.x - other.x) > EPSILON:
            return False
        if abs(self.y - other.y) > EPSILON:
            return False
        return True

    def __bool__(self):
        return bool(self.length2())

    def distance(self, other):
        return (self - other).length()

    def distance2(self, other):
        return (self - other).length2()

    def toList(self):
        return [self.x, self.y]


class Vec3:
    """
    Vector class with an X, Y and Z component.
    """

    def __init__(self, xx=None, yy=None, zz=None):
        """
                Initialize Vec3 object.

                If xx or (yy and zz) is not filled in, they will be initialized as follows:

                >>> Vec3() == Vec3(0, 0, 0)
                True
                >>> Vec3(1) == Vec3(1, 1, 1)
                True
                >>> Vec3(2, 3, 4) == Vec3(2, 3, 4)
                True

                :param xx: X coordinate of the vector.
                :param yy: Y coordinate of the vector.
                :param zz: Z coordinate of the vector.
                """
        if xx is None:
            self.x = 0
            self.y = 0
            self.z = 0

        elif isinstance(xx, Vec3):
            self.x = xx.x
            self.y = xx.y
            self.z = xx.z

        elif yy is None:
            self.x = xx
            self.y = xx
            self.z = xx
        else:
            self.x = xx
            self.y = yy
            self.z = zz

    def normalize(self):
        """
        Normalize self.

        This will set the length of self to 1.

        >>> v = Vec3(2, 0, 0)
        >>> v.normalize()
        >>> v == Vec3(1, 0, 0)
        True

        :return: None
        """
        length = self.length()
        if length > 0:
            self.x /= length
            self.y /= length
            self.z /= length

    def unit(self):
        """
        Return normalized version of self.

        >>> v = Vec3(0, 4, 0)
        >>> v.unit() == Vec3(0, 1, 0)
        True

        :return: normalized version of self.
        """
        length = self.length()
        if length > 0:
            return Vec3(self.x / length, self.y / length, self.z / length)
        return Vec3(self)

    def __mul__(self, other):
        """
        Element-wise Multiply self * other.

        Can work for other Vec3 or scalar

        >>> Vec3(2, 4, 3) * Vec3(3, 2, 1) == Vec3(6, 8, 3)
        True
        >>> Vec3(5, 3, 4) * 2 == Vec3(10, 6, 8)
        True

        :param other: multiplicand
        :return: self * other
        """
        if isinstance(other, numbers.Number):
            return Vec3(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    @staticmethod
    def random_uv():
        a = 1
        b = 1
        c = 1

        while a + b + c > 1:
            a = random()
            b = random()
            c = random()

        return Vec3(a,b,c)

    def dot(self, other):
        """
        Return dot product of self and other.

        >>> Vec3(3, 4, 2).dot(Vec3(2, 3, 1))
        20

        :param other: multiplicand
        :return: self dot other
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __bool__(self):
        return bool(self.length2())

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return Vec3(self.x / other, self.y / other, self.z / other)
        else:
            raise ValueError("Cant div by Vec3")

    def length2(self):
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def length(self):
        return sqrt(self.length2())

    def cross_product(self, other):
        return Vec3(self.y * other.z - other.y * self.z,
                    self.z * other.x - other.z * self.x,
                    self.x * other.y - other.x * self.y)

    def rotate(self, rotation):
        if abs(rotation.x) > rotation.y:
            Nt = Vec3(rotation.z, 0, -rotation.x) / sqrt(rotation.x ** 2 + rotation.z ** 2)
        else:
            Nt = Vec3(0, -rotation.z, rotation.y) / sqrt(rotation.y ** 2 + rotation.z ** 2)

        Nb = rotation.cross_product(Nt)

        x = self.x * Nb.x + self.y * rotation.x + self.z * Nt.x
        y = self.x * Nb.y + self.y * rotation.y + self.z * Nt.y
        z = self.x * Nb.z + self.y * rotation.z + self.z * Nt.z

        self.x = x
        self.y = y
        self.z = z

        self.normalize()

    def rotated(self, rotation):
        if abs(rotation.x) > abs(rotation.y):
            Nt = Vec3(rotation.z, 0, -rotation.x) / sqrt(rotation.x ** 2 + rotation.z ** 2)
        else:
            Nt = Vec3(0, -rotation.z, rotation.y) / sqrt(rotation.y ** 2 + rotation.z ** 2)

        Nb = rotation.cross_product(Nt)

        x = self.x * Nb.x + self.y * rotation.x + self.z * Nt.x
        y = self.x * Nb.y + self.y * rotation.y + self.z * Nt.y
        z = self.x * Nb.z + self.y * rotation.z + self.z * Nt.z

        return Vec3(x, y, z)

    def distance(self, other):
        return (self - other).length()

    def distance2(self, other):
        return (self - other).length2()

    @staticmethod
    def point_on_hemisphere(normal=None):
        theta = random() * 2 * pi
        phi = acos(1 - 2 * random())

        if not normal:
            return Vec3(sin(phi) * cos(theta), abs(sin(phi) * sin(theta)), cos(phi))

        return Vec3.point_on_hemisphere().rotated(normal)

    @staticmethod
    def point_on_diffuse_hemisphere(normal=None):
        u = random()
        v = 2 * pi * random()

        if not normal:
            return Vec3(cos(v) * sqrt(u), sin(v) * sqrt(u), sqrt(1 - u))

        return Vec3.point_on_hemisphere().rotated(normal)

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        return "Vec3<{} {} {}>".format(self.x, self.y, self.z)

        # no support for this on the HPC 10 cluster
        # return f"Vec3<{self.x} {self.y} {self.z}>"

    def toList(self):
        return [self.x, self.y, self.z]

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __eq__(self, other):
        if abs(self.x - other.x) > EPSILON:
            return False
        if abs(self.y - other.y) > EPSILON:
            return False
        if abs(self.z - other.z) > EPSILON:
            return False
        return True

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def avg(self):
        return (self.x + self.y + self.z) / 3


def sigmoid(x):
    return 1/(1+(e**(-x)))


def sigmoid(x):
    return 1/(1+(e**(-x)))

def basic_tonemap(x):
    return x/(x+1)