import yaml


def set_visa_parameters():
    detect_contour_parameters = {'matcher': {'index_params': [1, 5], 'search_params': 70}, 'min_match_count': 10,
                                 'dst_threshold': 0.7, 'nfeatures': 200000, 'neighbours': 2}
    adjust_referee_colors_parameters = {'hsv_referee': {'low_h': 10, 'high_h': 180, 'low_v': 0, 'high_v': 100},
                                        'area_threshold': 100, 'coef': {'1': 0.8, '2': 1.2, '3': 0.95, '4': 1.02},
                                        'area_thre': 4}
    resize_banner_parameters = {'resize_coef': 6.8, 'w_threshold': 0.8}

    with open('visa_parameters.yml', 'w') as outfile:
        yaml.dump(detect_contour_parameters, outfile, default_flow_style=False)
        yaml.dump(adjust_referee_colors_parameters, outfile, default_flow_style=False)
        yaml.dump(resize_banner_parameters, outfile, default_flow_style=False)
