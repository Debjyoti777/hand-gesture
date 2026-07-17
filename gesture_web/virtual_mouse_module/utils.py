import math


def calculate_distance(point1, point2):

    x1 = point1.x
    y1 = point1.y

    x2 = point2.x
    y2 = point2.y

    distance = math.sqrt(
        (x2 - x1)**2 +
        (y2 - y1)**2
    )

    return distance