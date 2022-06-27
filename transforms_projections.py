from filling import *


def affine_transform(cp, theta=None, u=None, t=None):

    # M --> dimension of point | N --> number of points
    M, N = cp.shape
    
    # Initialize rotation and translation matrices to do nothing
    Rh = identity(M + 1)
    Th = identity(M + 1)

    # I need the identity matrix for computation purposes
    I = identity(M)

    # If a translation vector was given
    if t is not None:
        # Change Th based on 5.37
        Th = block([[I, t],
                    [zeros((1, M)), 1]])

    # If axis and angle were given
    if (u is not None) or (theta is not None):
        # Compute R based on 5.45
        R = (1 - cos(theta)) * outer(u, u) \
          + cos(theta) * I \
          + sin(theta) * cross(u, I, axis=0)
        
        # Change Rh based on 5.49
        Rh = block([[R, zeros((M, 1))],
                    [zeros((1, M)), 1]])
    
    # Get homogenous coordinates of cp
    cph = block([[cp],
                 [ones((1, N))]])

    # Transform cph to cqh
    cqh = (Th @ Rh) @ cph
    
    # Since the result is in homogenous form, return cq, not cqh
    return cqh[:M, :]


def system_transform(cp, R, c0):

    # M --> dimension of point | N --> number of points
    M, N = cp.shape

    # Given R, create the homogenous R^-1
    Rh = block([[R.T, -R.T @ c0],
                [zeros((1, M)), 1]])

    # Get homogenous coordinates of cp
    cph = block([[cp],
                 [ones((1, N))]])

    # Find coordinates of new system
    dh = Rh @ cph

    # Since the result is in homogenous form, return d, not dh
    return dh[:M, :]


def project_cam(f, cu, cx, cy, cz, p):
    
    # M --> dimension of point | N --> number of points
    M, N = p.shape

    # Create R from the camera's base vectors
    R = block([[cx, cy, cz]])

    # Find coordinates from the camera's point of view based on 6.13
    q = system_transform(p, R, cu)
    
    # Calculate the euclidean distance from the camera
    depth = linalg.norm(q, axis=0).reshape(1, -1)

    # Project all points to camera's panel using 
    z = q[M - 1, :]
    verts2d = q[:M - 1, :]
    verts2d *= (f / z)
    
    return verts2d, depth


def project_cam_lookat(f, c_org, c_lookat, c_up, verts3d):

    # Calculate camera's z using 6.6
    cz = (c_org - c_lookat) / linalg.norm(c_org - c_lookat)

    # Calculate camera's y using 6.7
    t = c_up - dot(c_up.T, cz) * cz
    cy = t / linalg.norm(t)

    # Calculate camera's x using 6.8
    cx = cross(cy, cz, axis=0)

    verts2d, depth = project_cam(f, c_org, cx, cy, cz, verts3d)

    return verts2d, depth


def rasterize(verts2d, img_h, img_w, cam_h, cam_w):
    
    # Move origin to bottom left of camera's panel
    verts2d += array([[cam_w / 2],
                      [cam_h / 2]])
    
    # Scale them to fit the image
    verts2d *= array([[img_w / cam_w],
                      [img_h / cam_h]])

    return verts2d.round()


def render_object(verts3d, faces, vcolors, img_h, img_w, cam_h, cam_w, f, c_org, c_lookat, c_up):

    # Find vertices and their depth based on camera's 
    verts2d, depth = project_cam_lookat(f, c_org, c_lookat, c_up, verts3d)
    
    # Transform from inches to pixels
    verts_rast = rasterize(verts2d, img_h, img_w, cam_h, cam_w)
    
    # Color triangles inside image
    I = render(verts_rast.T, faces, vcolors, depth.reshape(-1), 'gouraud', img_h, img_w)

    return I