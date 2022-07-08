from numpy import *
from numpy.linalg import norm


def ambient_light(ka, Ia):
    return ka * Ia


def diffuse_light(P, N, color, kd, light_positions, light_intensities):

    I = zeros(shape=(3, 1))

    for (source, intensity) in zip(light_positions, light_intensities):
        L = (source - P) / norm(source - P)
        if N.T @ L > 0:
            I += kd * (1 / (norm(source - P) ** 2)) * intensity * (N.T @ L)
    
    return I * color


def specular_light(P, N, color, cam_pos, ks, n, light_positions, light_intensities):

    I = zeros(shape=(3, 1))
    
    for (source, intensity) in zip(light_positions, light_intensities):

        L = (source - P) / norm(source - P)
        R = (2 * N.T @ L) * N - L
        V = (cam_pos - P) / norm(cam_pos - P)

        if V.T @ R > 0 and N.T @ L > 0:
            I += ks * (1 / (norm(source - P) ** 2)) * intensity * ((V.T @ R) ** n)
    
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