from numpy import *
from numpy.linalg import *
from filling import *
from transforms_projections import *


def ambient_light(ka, Ia):
    I = ka * Ia
    return I


def diffuse_light(P, N, color, kd, light_positions, light_intensities):

    I = zeros((3,))

    L = (light_positions - P) / norm(light_positions - P, axis=0)
    
    for i in range(light_positions.shape[1]):
        if N.T @ L[:, i] > 0:
            I += kd * light_intensities[:, i] * (N.T @ L[:, i])
    
    return I.reshape(-1, 1) * color


def specular_light(P, N, color, cam_pos, ks, n, light_positions, light_intensities):

    I = zeros((3,))

    L = (light_positions - P) / norm(light_positions - P, axis=0)
    R = (2 * N.T @ L) * N - L
    V = (cam_pos - P) / norm(cam_pos - P, axis=0)
    
    for i in range(light_positions.shape[1]):
        if V.T @ R[:, i] > 0 and N.T @ L[:, i] > 0:
            I += ks * light_intensities[:, i] * (V.T @ R[:, i]) ** n
    
    return I.reshape(-1, 1) * color


def calculate_normals(vertices, face_indices):

    normals = empty(shape=vertices.shape)

    surface_normals = cross(vertices[:, face_indices[1]] - vertices[:, face_indices[0]],
                            vertices[:, face_indices[2]] - vertices[:, face_indices[0]],
                            axis=0)

    surface_normals /= norm(surface_normals, axis=0)

    for i in range(vertices.shape[1]):
        S_Nk = sum(surface_normals[:, where(face_indices == i)[1]], axis=1).reshape(-1, 1)
        S_Nk /= norm(S_Nk, axis=0)
        normals[:, i] = S_Nk.reshape(-1)
    
    return normals


def render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, face_indices, ka, kd, ks, n, light_position, light_intensities, Ia):
    
    # 1. Calculate surface normals of all vertices of object
    normals = calculate_normals(verts, face_indices)
    
    # 2a. Change coordinate system based on where the camera is
    verts2d, depth = project_cam_lookat(f=focal,
                                        c_org=eye,
                                        c_lookat=lookat,
                                        c_up=up,
                                        verts3d=verts)

    # 2b. Project them in camera lens
    verts_rast = rasterize(verts2d=verts2d,
                           img_h=M,
                           img_w=N,
                           cam_h=H,
                           cam_w=W)
    
    I = render(verts2d=verts_rast,
               faces=face_indices,
               vcolors=vert_colors,
               depth=depth,
               shade_t=shader,
               img_h=M,
               img_w=N)

    pass


def shade_gouraud(verts_p, verts_n, verts_c, bcoords, cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, X):
    
    # Define somewhere to store the vertices' colors after the illumination model
    new_colors = empty(shape=verts_c.shape)

    # Apply illumination model to each vertex of the triangle
    for i in range(verts_c.shape[1]):

        vertex_color = verts_c[:, i] + \
            ambient_light(ka, Ia) + \
                diffuse_light(P=bcoords,
                              N=verts_n[:, i],
                              color=verts_c[:, i],
                              kd=kd,
                              light_positions=light_positions,
                              light_intensities=light_intensities) + \
                    specular_light(P=bcoords,
                                   N=verts_n[:, i],
                                   color=verts_c[:, i],
                                   cam_pos=cam_pos,
                                   ks=ks,
                                   n=n,
                                   light_positions=light_positions,
                                   light_intensities=light_intensities)
        
        new_colors[:, i] = vertex_color
        
        Y = shade_triangle(X, verts_p, new_colors, 'gouraud')
        # With verts_p, new_colors, 
    pass


def shade_phong(verts_p, verts_n, verts_c, bcoords, cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, X):
    pass