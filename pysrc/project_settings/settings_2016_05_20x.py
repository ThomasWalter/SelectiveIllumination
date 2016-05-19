
data_folder = '/Users/twalter/data/Perrine/data_May_2016/20x'
result_folder = '/Users/twalter/data/Perrine/results_May_20x'

img_debug_folder = os.path.join(result_folder, 'img_debug')
img_graph_overlay_folder = os.path.join(result_folder, 'img_graph_overlay') 
img_single_output_folder = os.path.join(result_folder, 'img_chosen_cells')
coordinate_folder = os.path.join(result_folder, 'coordinates')
metamorph_folder = os.path.join(result_folder, 'metamorph')

make_folder = [img_debug_folder,
               img_graph_overlay_folder, 
               img_single_output_folder, 
               coordinate_folder,
               metamorph_folder]

img_debug_folder_sub = 'img_debug'
img_graph_overlay_folder_sub = 'img_graph_overlay'
img_single_output_folder_sub = 'img_chosen_cells'
coordinate_folder_sub = 'coordinates'
metamorph_folder_sub = 'metamorph'

    
debug = False
debug_screen_output = True
graph_overlay = False
single_mask = True
coordinate_file = True
metamorph_export = True

param_pixel_size = 0.66
param_offset = 0.0
cluster_size = 1
cluster_dist = 1

# segmentation_folder = os.path.join(result_folder, 'segmentation')
# cell_selection_folder = os.path.join(result_folder, 'cell_selection')
# debug_folder = os.path.join(result_folder, 'debug')
# debug_prefilter_test = os.path.join(result_folder, 'prefilter_test')
#debug_graph_overlay = os.path.join(debug_folder, 'graph_overlay')

prefilter_settings = {
                      'median': {'median_size': 2,},
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

                      # These are the relevant settings for the prefiltering
                      'med_denbi_asfrec': {
                                           # prefiltering
                                           'median_size': 1,
                                           
                                           # weight of the gray level values
                                           # grey level values are weighted with a Gaussian
                                           # according to the distance to the center value. 
                                           'sigma_signal': .05,
                                           'sigma_space': 2,

                                           # bins and win_size are also parameters for the bilateral filter 
                                           # but only important for performance. 
                                           'bins': 64,  
                                           'win_size': 9, 
                                           
                                           # closing and opening by reconstruction                                           
                                           'close_size': 1,
                                           'open_size': 1, 
                                           
                                           },                      
                      
                      }


homogenize_settings = {'h': 10}

segmentation_settings = {
                         'prefiltering': 'med_denbi_asfrec',
                         'thresh': 2,
                         'thresh_hmax': 2,

                         #'bg_sub': 'constant_median',
                         'bg_sub': 'avg',    
                         #'bg_sub': 'med',         
                         'hole_size': 7,                                          
                         }

background_subtraction = {
                          'radius': 20,
                          }

postfilter = {'area': 50,              
              'mean_intensity': 5.0,

              # threshold for split lines: if the mean intensity of the gradient is lower than this parameter
              # then the split line is rejected. A value of 0.0 for instance is neutral (does not reject anything). 
              'wsl_mean_intensity': 0.0,
              }


split_cells = {
               'h': 2,
               'sigma' : 2.0,      
               }

# graph output
graph_radius = 5
graph_color_code = {0: (0, 200, 10), # node is chosen
                    1: (0, 10, 220), # node is not chosen, but allowed
                    2: (220, 10, 10), # node is not allowed
                    }




