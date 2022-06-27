from numpy import *


def load_data(file):
    return load(file, allow_pickle=True).tolist().values()


edge = [('y_min', int),
        ('y_max', int),
        ('slope', float),
        ('intersect', float),
        ('RGB_min', float, (3,)),
        ('RGB_max', float, (3,))]


def create_edge(vertices):

    info = (
        # Store lower vertex's y-coordinate.
        vertices[argmin(vertices[:, 1]), 1],
        # Store higher vertex's y-coordinate.
        vertices[argmax(vertices[:, 1]), 1],
        # If vertices have same x-coordinate, then the edge is vertical with
        # slope infinity. Else its y2 - y1 / x2 - x1.
        inf if vertices[0, 0] == vertices[1, 0] else (vertices[0, 1] - vertices[1, 1]) / (vertices[0, 0] - vertices[1, 0]),
        # Initiate intersect to be the x-coordinate of the lower vertex.
        vertices[argmin(vertices[:, 1]), 0],
        # Store lower vertex's RGB.
        vertices[argmin(vertices[:, 1]), 2:],
        # Store higher vertex's RGB.
        vertices[argmax(vertices[:, 1]), 2:]
    )

    return array(info, edge)


def interpolate_color(x1, x2, x, C1, C2):

    value = (((x1 - x) / (x1 - x2)).reshape(len(C2.shape), 1)) * C2 + (((x - x2) / (x1 - x2)).reshape(len(C1.shape), 1)) * C1
    return value


def shade_triangle(img, verts2d, vcolors, shade_t, img_h, img_w):
    # Each vertex needs its respective color for my edge structure.
    verts2d = append(verts2d, vcolors, axis=1)

    # Split the vertices into three tuples, for each edge to take.
    verts2d = array([delete(verts2d, i, 0) for i in range(3)])

    # For each tuple, create it's edge.
    triangle_edges = fromiter((create_edge(verts2d[i]) for i in range(3)), dtype=edge)

    # Scan only from the lowest point of the triangle, to its highest point.
    lowest_scanline = min(triangle_edges['y_min'])
    highest_scanline = max(triangle_edges['y_max'])

    # Active edges are those edges of the triangle that start from the lowest
    # scanline.
    active_edges = triangle_edges[(triangle_edges['y_min'] == lowest_scanline)]

    for y in range(lowest_scanline, highest_scanline):

        # Delete those edges that end in the current scanline.
        lower_edges = active_edges[active_edges['y_max'] == y]
        if lower_edges.size > 0:
            active_edges = delete(active_edges, active_edges == lower_edges)

        # Since I will be filling a line from left to right,
        # sort the edges based on intersect in ascending order.
        active_edges = sort(active_edges, order='intersect')

        # Fill from the leftmost to the rightmost intersect.
        leftmost_intersect = ceil(active_edges[0]['intersect'])
        rightmost_intersect = ceil(active_edges[1]['intersect'])

        # If flat, shade triangle with the mean color of its vertices.
        if shade_t == "flat":
            C = vcolors.mean(axis=0)

        # If not flat then gouraud, thus interpolate to find the color of the
        # intersects.
        # Cl --> left intersect, Cr --> right intersect.
        elif shade_t == "gouraud":
            Cl, Cr = interpolate_color(active_edges['y_max'],
                                       active_edges['y_min'],
                                       y,
                                       active_edges['RGB_max'],
                                       active_edges['RGB_min'])

        for x in range(int(leftmost_intersect), int(rightmost_intersect)):

            if (x in range(0, img_w)) and (y in range(0, img_h)):

                if shade_t == "flat":
                    # Just paint (x, y).
                    img[x, y] = C

                if shade_t == "gouraud":
                    # Knowing Ca and Cb, interpolate for (x, y) and paint it.
                    img[x, y] = interpolate_color(rightmost_intersect,
                                                  leftmost_intersect,
                                                  x,
                                                  Cr,
                                                  Cl)

        # Since I will be moving upwards, move every intersect,
        # one step up according to its slope.
        active_edges['intersect'] += 1 / active_edges['slope']

        # Add the edges that start from the next scanline.
        upper_edges = triangle_edges[triangle_edges['y_min'] == y + 1]
        if upper_edges.size > 0:
            active_edges = append(active_edges, upper_edges)

    return img


def render(verts2d, faces, vcolors, depth, shade_t, img_h, img_w):

    # Colors are given BGR instead of RGB. Flip them.
    vcolors = flip(vcolors, axis=1)
    # Also, since they range from 0-1, change it to 0-255.
    vcolors *= 255

    # Vertices are given (y, x) instead of (x, y). Flip them.
    verts2d = flip(verts2d, axis=1)

    # Calculate the depth of all the triangles.
    depth_of_triangles = sum(depth[faces], axis=1, dtype=float) / 3

    # Sort them out based on highest depth first.
    priority_of_triangles = flip(argsort(depth_of_triangles))

    # Create the canvas and scale it to 255.
    img = 255 * ones((img_h, img_w, 3))

    # With the priority in mind, shade a triangle one by one.
    for face in faces[priority_of_triangles]:
        img = shade_triangle(img, verts2d[face], vcolors[face], shade_t, img_h, img_w)
    return img
