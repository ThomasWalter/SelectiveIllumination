
data_folder = '/Users/twalter/data/Perrine/max_proj'
result_folder = '/Users/twalter/data/Perrine/results'

segmentation_folder = os.path.join(result_folder, 'segmentation')
cell_selection_folder = os.path.join(result_folder, 'cell_selection')
debug_folder = os.path.join(result_folder, 'debug')
debug_prefilter_test = os.path.join(result_folder, 'prefilter_test')
debug_graph_overlay = os.path.join(debug_folder, 'graph_overlay')

prefilter_settings = {'median': {'median_size': 2,},
                      'avg': {'avg_size': 2,},
                      'bilateral': {'bil_radius': 3, 'bil_lower': 10, 'bil_upper': 20,},
                      'denoise_bilateral': {'win_size': 9, 
                                            'sigma_signal': .15,
                                            'sigma_space': 5,
                                            'bins': 64,},
                      'close_rec': {'close_size': 3},  
                      'denbi_clorec': {'win_size': 9, 
                                       'sigma_signal': .15,
                                       'sigma_space': 5,
                                       'bins': 64,  
                                       'close_size': 5},                        
                      'denbi_asfrec': {'win_size': 9, 
                                       'sigma_signal': .15,
                                       'sigma_space': 5,
                                       'bins': 64,  
                                       'close_size': 5,
                                       'open_size': 5},
                      'med_denbi_asfrec': {'win_size': 9, 
                                       'sigma_signal': .15,
                                       'sigma_space': 5,
                                       'bins': 64,  
                                       'close_size': 5,
                                       'open_size': 5, 
                                       'median_size': 1},
                      }


homogenize_settings = {'h': 100}

segmentation_settings = {
                         'prefiltering': 'med_denbi_asfrec',
                         'thresh': 2,
                         'thresh_hmax': 10,

                         #'bg_sub': 'constant_median',
                         'bg_sub': 'avg',    
                         #'bg_sub': 'med',                                                   
                         }

background_subtraction = {
                          'radius': 60,
                          }

postfilter = {'area': 500,
              'mean_intensity': 10,
              'wsl_mean_intensity': 3.0,
              }


split_cells = {
               'h': 1,
               'sigma' : 5.0,               
               }

# graph output
graph_radius = 6
graph_color_code = {0: (0, 200, 10), # node is chosen
                    1: (0, 10, 220), # node is not chosen, but allowed
                    2: (220, 10, 10), # node is not allowed
                    }


debug = True
make_folder = [segmentation_folder,
               cell_selection_folder, 
               debug_folder, 
               debug_prefilter_test,
               debug_graph_overlay]

