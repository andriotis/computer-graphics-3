from illumination import *
from cv2 import imwrite

if __name__ == '__main__':

    model = load_data_properly('h3.npy')
    shaders = ['gouraud', 'phong']
    
    for (name, params) in model:
        for shader in shaders:
            img = render_object(shader,70, params['cam_eye'], params['cam_lookat'],
                                params['cam_up'], params['bg_color'], params['M'],
                                params['N'], params['H'], params['W'],
                                params['verts'], params['vertex_colors'],
                                params['face_indices'], params['ka'], params['kd'],
                                params['ks'], params['n'], params['light_positions'],
                                params['light_intensities'], params['Ia'])

            imwrite(f'./{shader}_{name}.png', 255 * img)