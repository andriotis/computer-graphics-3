from numpy import *
from numpy.linalg import *
from filling import *
from transforms_projections import *


def ambient_light(ka, Ia):
    return ka * Ia


def diffuse_light(P, N, color, kd, light_positions, light_intensities):

    I = zeros(shape=(3, 1))

    for (source, intensity) in zip(light_positions, light_intensities):
        L = (source - P) / norm(source - P)
        if N.T @ L > 0:
            I += kd * intensity * (N.T @ L)
    
    return I * color


def specular_light(P, N, color, cam_pos, ks, n, light_positions, light_intensities):

    I = zeros(shape=(3, 1))
    
    for (source, intensity) in zip(light_positions, light_intensities):

        L = (source - P) / norm(source - P)
        V = (cam_pos - P) / norm(cam_pos - P)
        R = (2 * N.T @ L) * N - L

        if V.T @ R > 0 and N.T @ L > 0:
            I += ks * intensity * ((V.T @ R) ** n)
    
    return I * color


def calculate_normals(vertices, face_indices):
    
    normals = empty(shape=vertices.shape)

    surface_normals = cross(vertices[:, face_indices[1]] - vertices[:, face_indices[0]],
                            vertices[:, face_indices[2]] - vertices[:, face_indices[0]],
                            axis=0)

    surface_normals /= norm(surface_normals, axis=0)

    for i in range(vertices.shape[1]):
        S_Nk = sum(surface_normals[:, where(face_indices == i)[1]], axis=1).reshape(3, 1)
        S_Nk /= norm(S_Nk)
        normals[:, i] = S_Nk.reshape(-1)
    
    return normals


def render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia):

    normals = calculate_normals(vertices=verts, face_indices=face_indices)

    verts2d, depth = project_cam_lookat(f=focal, c_org=eye, c_lookat=lookat, c_up=up, verts3d=verts)

    verts_rast = rasterize(verts2d=verts2d, img_h=N, img_w=M, cam_h=H, cam_w=W)
    
    vert_colors = flip(vert_colors, axis=0)
    
    verts_rast = flip(verts_rast, axis=0)

    depth = depth.reshape(-1)

    depth_of_triangles = mean(depth[face_indices], axis=0)

    priority_of_triangles = flip(argsort(depth_of_triangles))

    X = ones((M, N, 3)) * bg_color.reshape(-1)

    for triangle in priority_of_triangles:

        if (all(verts_rast[:, face_indices[:, triangle]][0, :]) < W) and (all(verts_rast[:, face_indices[:, triangle]][1, :]) < H):

            X = globals()[f'shade_{shader}'](verts_p=verts_rast[:, face_indices[:, triangle]],
                                             verts_n=normals[:, face_indices[:, triangle]],
                                             verts_c=vert_colors[:, face_indices[:, triangle]],
                                             bcoords=mean(verts_rast[:, face_indices[:, triangle]], axis=0).reshape(-1, 1),
                                             cam_pos=eye,
                                             ka=ka,
                                             kd=kd,
                                             ks=ks,
                                             n=n,
                                             light_positions=light_positions,
                                             light_intensities=light_intensities,
                                             Ia=Ia,
                                             X=X)

    return X


def shade_gouraud(verts_p, verts_n, verts_c, bcoords, cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, X):

    for i in range(3):

        I_amb = ambient_light(ka, Ia)

        I_diff = diffuse_light(P=bcoords,
                               N=verts_n[:, i].reshape(3, 1),
                               color=verts_c[:, i].reshape(3, 1),
                               kd=kd,
                               light_positions=light_positions,
                               light_intensities=light_intensities)

        I_spec = specular_light(P=bcoords,
                                N=verts_n[:, i].reshape(3, 1),
                                color=verts_c[:, i].reshape(3, 1),
                                cam_pos=cam_pos.reshape(3, 1),
                                ks=ks,
                                n=n,
                                light_positions=light_positions,
                                light_intensities=light_intensities)
        
        verts_c[:, i] = (I_amb + I_diff + I_spec).reshape(-1)
    
    verts_pcn = hstack((verts_p.T, verts_c.T, verts_n.T))

    tuples_of_verts = array([delete(verts_pcn, i, axis=0) for i in range(3)])

    triangle_edges = fromiter((create_edge(tuples_of_verts[i]) for i in range(3)), dtype=edge)

    lowest_scanline = min(triangle_edges['y_min'])
    highest_scanline = max(triangle_edges['y_max'])

    active_edges = triangle_edges[(triangle_edges['y_min'] == lowest_scanline)]

    for y in range(lowest_scanline, highest_scanline):

        lower_edges = active_edges[active_edges['y_max'] == y]
        if lower_edges.size > 0:
            active_edges = delete(active_edges, active_edges == lower_edges)

        active_edges = sort(active_edges, order='intersect')

        leftmost_intersect = ceil(active_edges[0]['intersect'])
        rightmost_intersect = ceil(active_edges[1]['intersect'])

        Cl, Cr = interpolate_color(active_edges['y_max'],
                                   active_edges['y_min'],
                                   y,
                                   active_edges['RGB_max'],
                                   active_edges['RGB_min'])

        for x in range(int(leftmost_intersect), int(rightmost_intersect)):

            X[x, y] = interpolate_color(rightmost_intersect,
                                        leftmost_intersect,
                                        x,
                                        Cr,
                                        Cl)

        active_edges['intersect'] += 1 / active_edges['slope']

        upper_edges = triangle_edges[triangle_edges['y_min'] == y + 1]
        if upper_edges.size > 0:
            active_edges = append(active_edges, upper_edges)

    return X


def shade_phong(verts_p, verts_n, verts_c, bcoords, cam_pos, ka, kd, ks, n, light_positions, light_intensities, Ia, X):
        
    verts_pcn = hstack((verts_p.T, verts_c.T, verts_n.T))

    tuples_of_verts = array([delete(verts_pcn, i, axis=0) for i in range(3)])

    triangle_edges = fromiter((create_edge(tuples_of_verts[i]) for i in range(3)), dtype=edge)

    lowest_scanline = min(triangle_edges['y_min'])
    highest_scanline = max(triangle_edges['y_max'])

    active_edges = triangle_edges[(triangle_edges['y_min'] == lowest_scanline)]

    for y in range(lowest_scanline, highest_scanline):

        lower_edges = active_edges[active_edges['y_max'] == y]
        if lower_edges.size > 0:
            active_edges = delete(active_edges, active_edges == lower_edges)

        active_edges = sort(active_edges, order='intersect')

        leftmost_intersect = ceil(active_edges[0]['intersect'])
        rightmost_intersect = ceil(active_edges[1]['intersect'])

        Cl, Cr = interpolate_color(active_edges['y_max'],
                                   active_edges['y_min'],
                                   y,
                                   active_edges['RGB_max'],
                                   active_edges['RGB_min'])

        Nl, Nr = interpolate_color(active_edges['y_max'],
                                   active_edges['y_min'],
                                   y,
                                   active_edges['N_max'],
                                   active_edges['N_min'])


        for x in range(int(leftmost_intersect), int(rightmost_intersect)):

            color = interpolate_color(rightmost_intersect,
                                      leftmost_intersect,
                                      x,
                                      Cr,
                                      Cl).reshape(3, 1)
            
            normal = interpolate_color(rightmost_intersect,
                                      leftmost_intersect,
                                      x,
                                      Nr,
                                      Nl).reshape(3, 1)

            I_amb = ambient_light(ka, Ia)

            I_diff = diffuse_light(P=bcoords,
                                   N=normal,
                                   color=color,
                                   kd=kd,
                                   light_positions=light_positions,
                                   light_intensities=light_intensities)

            I_spec = specular_light(P=bcoords,
                                    N=normal,
                                    color=color,
                                    cam_pos=cam_pos,
                                    ks=ks,
                                    n=n,
                                    light_positions=light_positions,
                                    light_intensities=light_intensities)

            color = I_amb + I_diff + I_spec

            X[x, y] = color.reshape(-1)

        active_edges['intersect'] += 1 / active_edges['slope']

        upper_edges = triangle_edges[triangle_edges['y_min'] == y + 1]
        if upper_edges.size > 0:
            active_edges = append(active_edges, upper_edges)

    return X