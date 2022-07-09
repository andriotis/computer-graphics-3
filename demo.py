from numpy import load

if __name__ == '__main__':
    for (key, value) in load('h3.npy', allow_pickle=True).tolist().items():
        exec(key + ' = value')

    verts = verts.T
    vertex_colors = vertex_colors.T
    face_indices = face_indices.T
    cam_eye = cam_eye.reshape(3, 1)
    cam_up = cam_up.reshape(3, 1)
    cam_lookat = cam_lookat.reshape(3, 1)
    light_positions[0] = light_positions[0].reshape(3, 1)
    light_intensities[0] = light_intensities[0].reshape(3, 1)
    bg_color = bg_color.reshape(3, 1)
    Ia = Ia.reshape(3, 1)
    print('hello')