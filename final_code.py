from IPython.display import clear_output
from time import sleep
import warnings
from matplotlib import pyplot as plt, animation
from math import *
from MathUtil import *
import matplotlib

# ThreadPool works in interactive environments, but Pool is faster
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool

matplotlib.use("TkAgg")


class PointLight:
    def __init__(self, pos, color, size):
        '''
        Point light to make your world a little bit happier.
        :param pos: position of the light
        :param color: the color of the light
        :param size: the size of the light
        '''
        self.pos = pos
        self.offset = Vec3()
        self.color = color
        self.size = size

    def move_light(self):
        '''
        Generate a random offset for this light.
        The larger the light size, the larger the offsets
        '''
        self.offset = Vec3((random() - 0.5) * self.size,
                           (random() - 0.5) * self.size,
                           (random() - 0.5) * self.size)

    def get_pos(self):
        '''
        Gets the position of the light source.
        The possition is translated by the offset vector
        :return: Light source position
        '''
        return self.pos + self.offset


class AABB:
    def __init__(self, min_corner, max_corner):
        '''
        Axis aligned bounding box to contain all the triangles in the scene.
        If a ray does not hit this box, it will also miss all triangles.
        :param min_corner: vector discribing the corner with the lowest x,y and z values
        :param max_corner: vector discribing the corner with the highest x,y and z values
        '''
        self.min_corner = min_corner
        self.max_corner = max_corner


class Triangle:
    def __init__(self, a, b, c, Kd, Ks, Ns, refl):
        '''
        Basic triangle class.
        :param a: First corner
        :param b: Second corner
        :param c: Third corner
        :param Kd: Diffuse reflectivity
        :param Ks: Specular reflectivity
        :param Ns: Shininess
        :param refl: mirror like reflectivity
        '''
        self.a = a
        self.b = b
        self.c = c
        self.Kd = Kd
        self.Ks = Ks
        self.Ns = Ns
        self.refl = refl

        self.normal = (b - a).cross_product(c - a).unit()


class Sphere:
    def __init__(self, center, radius, Kd, Ks, Ns, refl):
        '''
        Basic sphere class
        :param center: center of the sphere
        :param radius: radius of the sphere
        :param Kd: Diffuse reflectivity
        :param Ks: Specular reflectivity
        :param Ns: Shininess
        :param refl: mirror like reflectivity
        '''
        self.center = center
        self.radius = radius
        self.Kd = Kd
        self.Ks = Ks
        self.Ns = Ns
        self.refl = refl


class Ray:
    def __init__(self, origin, direction):
        '''
        Basic ray class.
        :param origin: Origin/starting point of the ray
        :param direction: Direction of the ray
        '''
        self.origin = origin
        self.direction = direction

    def ray_AABB_intersect(self, other):
        # Find out what the first and last time is where the ray has the correct x coordinate
        tmin = (other.min_corner.x - self.origin.x) / self.direction.x
        tmax = (other.max_corner.x - self.origin.x) / self.direction.x

        if tmin > tmax:
            tmin, tmax = tmax, tmin

        # Also Find out what the first and last time is where the ray has the correct y coordinate
        tymin = (other.min_corner.y - self.origin.y) / self.direction.y
        tymax = (other.max_corner.y - self.origin.y) / self.direction.y

        if tymin > tymax:
            tymin, tymax = tymax, tymin

        # If the first time the x coordinate is correct is later than the last time the y coordinate is correct,
        # there is no hit.
        # There is also no hit first time the y coordinate is correct happens after the last time the x is correct
        if (tmin > tymax) or (tymin > tmax):
            return False

        # Now we want to know the window of time where both the x and y are correct
        if tymin > tmin:
            tmin = tymin

        if tymax < tmax:
            tmax = tymax

        # Calculate the first and last time the ray has the correct z coordinate
        tzmin = (other.min_corner.z - self.origin.z) / self.direction.z
        tzmax = (other.max_corner.z - self.origin.z) / self.direction.z

        if tzmin > tzmax:
            tzmin, tzmax = tzmax, tzmin

        # If the first time the z coordinate is correct happens after the window where x and y are correct,
        # there is no hit.
        # There is also no hit if the last time the z coordinate is correct happens before the x and y are correct
        if (tmin > tzmax) or (tzmin > tmax):
            return False

        # If there is a point on the ray where all the coordinates are inside the AABB, there is a hit!
        return True

    def sphere_intersection_test(self, other):
        B_min_C = self.origin - other.center
        a = self.direction.dot(self.direction)
        b = 2 * B_min_C.dot(self.direction)
        c = B_min_C.dot(B_min_C) - other.radius * other.radius

        # Caluculate the discriminant
        discriminant = b ** 2 - 4 * a * c

        # If the discriminant is smaller than 0, there is no hit.
        if discriminant < 0:
            return False
        else:
            # Calculate the hit location
            t = (-b - sqrt(discriminant)) / (2 * a)
            if t < 0:
                return False
            return t

    def triangle_intersection_test(self, other):
        '''
        Test if the ray hits the triangle.
        :param self: The ray
        :param other: The triangle
        :return: Distance to triangle, 0/False is no hit
        '''

        # Calculate the vectors describing the edges of the triangle
        edge1 = other.b - other.a
        edge2 = other.c - other.a

        # Check if the ray and triangle are not parallel
        h = self.direction.cross_product(edge2)
        a = edge1.dot(h)

        if -EPSILON < a < EPSILON:
            return False

        # Calculate U/V (barycentric coordinates) of the hit on the triangle
        f = 1.0 / a
        s = self.origin - other.a
        u = f * (s.dot(h))

        q = s.cross_product(edge1)
        v = f * self.direction.dot(q)

        # The U must be between 0 and one
        if u < 0 or u > 1:
            return False

        # The V must be larger than 0, and U+V must be smaller than one
        if v < 0 or u + v > 1:
            return False

        # Calculate the distance from the origin of the ray to the hit
        t = f * edge2.dot(q)

        # The distance must be positive because,
        # otherwise the triangle could be behind the ray
        if t < EPSILON:
            return False

        # There was a hit! Return the distance
        return t


def specular(hit_object, posHit, lightPos, cameraPos, normal):
    '''
    Calculate the fraction of specularly reflected light.
    :param triangle: The triangle that has been hit
    :param posHit: The position where it was hit
    :param lightPos: The position of the light source
    :param cameraPos: The position of the camera
    :param normal: The surface normal direction at the point that was hit
    :return: The fraction of incoming light that is specularly reflected
    '''
    lightDirection = (lightPos - posHit).unit()
    reflec = (2 * (normal.dot(lightDirection)) * normal - lightDirection)
    spec = max((cameraPos - posHit).unit().dot(reflec), 0)
    return spec ** hit_object.Ns * hit_object.Ks


def diffuse(hit_object, posHit, lightPos, normal):
    '''
    Calculate the fraction of diffusely reflected light.
    :param triangle: The triangle that has been hit
    :param posHit: The position where it was hit
    :param lightPos: The position of the light source
    :param normal: The surface normal direction at the point that was hit
    :return: The fraction of incoming light that is diffusely reflected
    '''
    lightDirection = (lightPos - posHit).unit()
    return max(lightDirection.dot(normal), 0) * hit_object.Kd


def trace_ray(ray, triangles, spheres, lights, bounding_box):
    '''
    Returns the color of the ray.
    Color changes depending on the distance to the hit
    :param ray: Ray that has been shot into the world
    :param triangles: list of triangles in the world
    :param lights: list of lights in the world
    :return: color
    '''

    if not ray.ray_AABB_intersect(bounding_box):
        return Vec3(0.0)

    has_hit = False
    closest_hit = inf
    closest_obj = None
    normal = None

    for tri in triangles:
        t = ray.triangle_intersection_test(tri)
        if t:
            has_hit = True
            if t < closest_hit:
                closest_hit = t
                closest_obj = tri
                normal = tri.normal

    # Check if a shpere is hit
    for sph in spheres:
        t = ray.sphere_intersection_test(sph)
        if t:
            has_hit = True
            if t < closest_hit:
                closest_hit = t
                closest_obj = sph
                normal = (ray.origin + ray.direction * (closest_hit - EPSILON) - sph.center).unit()

    if has_hit:
        hit_pos = ray.origin + ray.direction * (closest_hit - EPSILON)

        col = Vec3(0.0)
        for light in lights:
            light.move_light()
            if check_if_visible(hit_pos, light.get_pos(), closest_obj, triangles, spheres):
                col += diffuse(closest_obj, hit_pos, light.get_pos(), normal) * light.color
                col += specular(closest_obj, hit_pos, light.get_pos(), ray.origin, normal) * light.color

        if closest_obj.refl > 0:
            reflec_direction = (ray.direction - 2 * (normal.dot(ray.direction)) * normal)

            bounce_ray = Ray(hit_pos, reflec_direction)

            col += trace_ray(bounce_ray, triangles, spheres, lights, bounding_box) * closest_obj.refl

        return col
    # Else, return black
    return Vec3(0.0)


def check_if_visible(pos, light_pos, hit_object, triangles, spheres):
    '''
    Check if the point on the triangle is visible from the light source.
    :param pos: Hit point
    :param light_pos: Light source position
    :param hit_object: The object that is looked at
    :param triangles: All triangles in the world
    :param spheres: All the spheres in the world
    :return: Is the triangle visible
    '''
    direction = (pos - light_pos)
    direction.normalize()
    ray = Ray(light_pos, direction)

    has_hit = False
    closest_hit = inf
    closest_obj = None

    for tri in triangles:
        t = ray.triangle_intersection_test(tri)
        if t:
            has_hit = True
            if t < closest_hit:
                closest_hit = t
                closest_obj = tri

    # Loop over all the spheres
    for sph in spheres:
        t = ray.sphere_intersection_test(sph)
        if t:
            has_hit = True
            if t < closest_hit:
                closest_hit = t
                closest_obj = sph

    if has_hit:
        return closest_obj == hit_object
    return False


def create_aabb(triangles, spheres):
    x_min = y_min = z_min = inf
    x_max = y_max = z_max = -inf
    if triangles:
        x_min = min(x_min, min(min(triangle.a.x, triangle.b.x, triangle.c.x) for triangle in triangles))
        y_min = min(y_min, min(min(triangle.a.y, triangle.b.y, triangle.c.y) for triangle in triangles))
        z_min = min(z_min, min(min(triangle.a.z, triangle.b.z, triangle.c.z) for triangle in triangles))

        x_max = max(x_max, max(max(triangle.a.x, triangle.b.x, triangle.c.x) for triangle in triangles))
        y_max = max(y_max, max(max(triangle.a.y, triangle.b.y, triangle.c.y) for triangle in triangles))
        z_max = max(z_max, max(max(triangle.a.z, triangle.b.z, triangle.c.z) for triangle in triangles))

    if spheres:
        x_min = min(x_min, min(s.center.x - s.radius for s in spheres))
        y_min = min(y_min, min(s.center.y - s.radius for s in spheres))
        z_min = min(z_min, min(s.center.z - s.radius for s in spheres))

        x_max = max(x_max, min(s.center.x + s.radius for s in spheres))
        y_max = max(y_max, min(s.center.y + s.radius for s in spheres))
        z_max = max(z_max, min(s.center.z + s.radius for s in spheres))

    min_corner = Vec3(x_min, y_min, z_min)
    max_corner = Vec3(x_max, y_max, z_max)

    return AABB(min_corner, max_corner)


def render_row(data):
    '''
    Renders a row of width pixels, of the world of triangles.
    :param data: contains all the information needed to render a row
    '''
    has_warned_clip = False
    row = []
    for x in range(data["width"]):
        col = Vec3(0.0)

        for _ in range(SAMPLES_PER_PIXEL):
            xdir = (2 * (x + random()) * data["inv_width"] - 1) * data["angle"] * data["aspect_ratio"]
            ydir = (1 - 2 * (data["y"] + random()) * data["inv_height"]) * data["angle"]

            raydir = Vec3(xdir, ydir, 1)
            raydir.normalize()
            ray = Ray(data["camera_pos"], raydir)

            # Add the spheres
            col += trace_ray(ray, data["triangles"], data["spheres"], data["lights"], data["bounding_box"])

        col /= SAMPLES_PER_PIXEL
        col = col.toList()
        clipped_col = clip(col, 0.0, 1.0)
        if not has_warned_clip:
            if col != clipped_col:
                warnings.warn("Image is clipping! Lights might be to bright..")
                has_warned_clip = True
        row.append(clipped_col)
    return row


def camera(camera_pos, width, height, fov, triangles, spheres, lights):
    '''
    Renders a image of width*height pixels, of the world of triangles.
    As seen from the camera position
    The field of view is how "wide" the lens is.
    :param camera_pos: camera position
    :param width: horizontal pixels
    :param height: vertical pixels
    :param fov: field of view
    :param triangles: triangles in the world
    :param lights: list of lights in the world
    '''

    angle = tan(pi * 0.5 * fov / 180)
    aspect_ratio = width / height
    inv_width = 1 / width
    inv_height = 1 / height

    bounding_box = create_aabb(triangles, spheres)

    with Pool() as p:
        image = p.map(render_row, [{"angle": angle, "aspect_ratio": aspect_ratio,
                                    "inv_width": inv_width, "inv_height": inv_height,
                                    "bounding_box": bounding_box, "camera_pos": camera_pos,
                                    "triangles": triangles, "spheres": spheres,
                                    "lights": lights, "width": width, "y": y} for y in range(height)])
        # Instead of showing the image, just return it
    return image


SAMPLES_PER_PIXEL = 2
FRAMES = 60
DISTANCE = 3.5

if __name__ == '__main__':

    sphere_1 = Sphere(Vec3(0.5, -0.1, 4.3), 0.8, Kd=Vec3(0.4), Ks=Vec3(0.3), Ns=1, refl=0.8)
    sphere_2 = Sphere(Vec3(-0.5, -0, 2), 0.4, Kd=Vec3(0.3), Ks=Vec3(0.3), Ns=1, refl=0.8)
    spheres = [sphere_1, sphere_2]

    ground = [
        Triangle(Vec3(-0.9, -1, 2), Vec3(-0.9, -1, 4), Vec3(0.9, -1, 2), Kd=Vec3(1.0, 0.7, 0.1), Ks=Vec3(0.1, 0.1, 0.2),
                 Ns=1, refl=0.2),
        Triangle(Vec3(0.9, -1, 2), Vec3(-0.9, -1, 4), Vec3(0.9, -1, 4), Kd=Vec3(1.0, 0.7, 0.1), Ks=Vec3(0.1, 0.1, 0.2),
                 Ns=1, refl=0.2)]

    body = [Triangle(Vec3(-0.6, -0.9, 3), Vec3(-0.6, -0.3, 3), Vec3(0.3, -0.9, 3), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.3, -0.9, 3), Vec3(-0.6, -0.3, 3), Vec3(0.3, -0.3, 3), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.3, -0.9, 3), Vec3(0.3, -0.6, 3), Vec3(0.6, -0.9, 3), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.6, -0.9, 3), Vec3(0.3, -0.6, 3), Vec3(0.6, -0.6, 3), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),

            Triangle(Vec3(-0.6, -0.9, 3.5), Vec3(-0.6, -0.3, 3.5), Vec3(0.3, -0.9, 3.5), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.3, -0.9, 3.5), Vec3(-0.6, -0.3, 3.5), Vec3(0.3, -0.3, 3.5), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.3, -0.9, 3.5), Vec3(0.3, -0.6, 3.5), Vec3(0.6, -0.9, 3.5), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.6, -0.9, 3.5), Vec3(0.3, -0.6, 3.5), Vec3(0.6, -0.6, 3.5), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),

            Triangle(Vec3(-0.6, -0.3, 3), Vec3(-0.6, -0.3, 3.5), Vec3(0.3, -0.3, 3.5), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(-0.6, -0.3, 3), Vec3(0.3, -0.3, 3.5), Vec3(0.3, -0.3, 3), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.3, -0.3, 3), Vec3(0.3, -0.3, 3.5), Vec3(0.3, -0.6, 3.5), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.3, -0.3, 3), Vec3(0.3, -0.6, 3.5), Vec3(0.3, -0.6, 3), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.3, -0.6, 3), Vec3(0.3, -0.6, 3.5), Vec3(0.6, -0.6, 3.5), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.3, -0.6, 3), Vec3(0.6, -0.6, 3.5), Vec3(0.6, -0.6, 3), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.6, -0.6, 3), Vec3(0.6, -0.6, 3.5), Vec3(0.6, -0.9, 3), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(0.6, -0.6, 3.5), Vec3(0.6, -0.9, 3.5), Vec3(0.6, -0.9, 3), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(-0.6, -0.3, 3.5), Vec3(-0.6, -0.3, 3), Vec3(-0.6, -0.9, 3), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0),
            Triangle(Vec3(-0.6, -0.3, 3.5), Vec3(-0.6, -0.9, 3), Vec3(-0.6, -0.9, 3.5), Kd=Vec3(0.6, 0.2, 0.2),
                     Ks=Vec3(0.4, 0.2, 0.2), Ns=1, refl=0)]

    wheels = [
        Triangle(Vec3(-0.4, -1, 3), Vec3(-0.4, -0.9, 3), Vec3(-0.3, -1, 3), Kd=Vec3(0.6, 0.7, 0.6),
                 Ks=Vec3(0.2, 0.2, 0.2),
                 Ns=1, refl=0),
        Triangle(Vec3(-0.3, -1, 3), Vec3(-0.4, -0.9, 3), Vec3(-0.3, -0.9, 3), Kd=Vec3(0.6, 0.7, 0.6),
                 Ks=Vec3(0.2, 0.2, 0.2), Ns=1, refl=0),
        Triangle(Vec3(0.3, -1, 3), Vec3(0.3, -0.9, 3), Vec3(0.4, -1, 3), Kd=Vec3(0.6, 0.7, 0.6), Ks=Vec3(0.2, 0.2, 0.2),
                 Ns=1, refl=0),
        Triangle(Vec3(0.4, -1, 3), Vec3(0.3, -0.9, 3), Vec3(0.4, -0.9, 3), Kd=Vec3(0.6, 0.7, 0.6),
                 Ks=Vec3(0.2, 0.2, 0.2),
                 Ns=1, refl=0)]

    triangles = ground + body + wheels

    lights = [PointLight(pos=Vec3(-0.3, -0.4, 1), color=Vec3(0.2), size=0.1),
              PointLight(pos=Vec3(-3, 10, -3), color=Vec3(0.4), size=0.1),
              PointLight(pos=Vec3(0), color=Vec3(0.3), size=0.1),
              PointLight(pos=Vec3(-1, 0, 10), color=Vec3(0.4), size=0.1)]

    frames = []

    print("STARTING RENDER")
    
    for x in range(FRAMES):
        frames.append(camera(Vec3(0, 0, x / FRAMES * DISTANCE), 350, 350, 60, triangles, spheres, lights))
        print(f"Done frame: {x + 1}/{FRAMES}")

    fig = plt.figure()
    im = plt.imshow(frames[0])


    def update(i):
        im.set_data(frames[i])
        return im,


    ani = animation.FuncAnimation(fig, update, FRAMES, interval=50, blit=True)
    plt.show()
    ani.save("hi_res_render.html")
