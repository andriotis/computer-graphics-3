from numpy import *

def load_data_properly(file):

    # Somewhere to store the properly processed data
    complete = {}

    # For every key-value pair, preprocess and store
    for (key, value) in load(file, allow_pickle=True).tolist().items():
        # If is a matrix
        if type(value) is ndarray:
            # Not a vector but a matrix
            if value.ndim is 2:
                # Store its transpose
                complete[key] = value.T
            # Make it a column vector before storing it
            else:
                complete[key] = value.reshape(-1, 1)
        # Change the list's content into column vectors
        elif type(value) is list:
            for i in range(len(value)):
                value[i] = value[i].reshape(-1, 1)
            complete[key] = value
        else:
            complete[key] = value

    # Ambient has kd=ks=0 but ka != 0
    ambient = complete.copy()
    ambient['kd'] = ambient['ks'] = 0

    # Diffuse has ka=ks=0 but kd != 0
    diffuse = complete.copy()
    diffuse['ka'] = diffuse['ks'] = 0

    # Specular has ka=kd=0 but ks != 0
    specular = complete.copy()
    specular['ka'] = specular['kd'] = 0
    
    # Return both the name and the params for png creation purposes
    model_name = ['ambient', 'diffuse', 'specular', 'complete']
    model_params = [ambient, diffuse, specular, complete]

    return zip(model_name, model_params)


edge = [('y_min', int),
        ('y_max', int),
        ('slope', float),
        ('intersect', float),
        ('RGB_min', float, (3,)),
        ('RGB_max', float, (3,)),
        ('N_min', float, (3,)),
        ('N_max', float, (3,))]


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
        vertices[argmin(vertices[:, 1]), 2:5],
        # Store higher vertex's RGB.
        vertices[argmax(vertices[:, 1]), 2:5],
        # Store lower vertex's surface normal
        vertices[argmin(vertices[:, 1]), 5:],
        # Store higher vertex's surface normal
        vertices[argmax(vertices[:, 1]), 5:],
    )

    return array(info, edge)


def interpolate_color(x1, x2, x, C1, C2):

    value = (((x1 - x) / (x1 - x2)).reshape(len(C2.shape), 1)) * C2 + (((x - x2) / (x1 - x2)).reshape(len(C1.shape), 1)) * C1
    return value
