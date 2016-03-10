"""
GUI for Cell Grid Illumination
"""

import os, sys, time

from Tkinter import *
from ttk import *

import tkFileDialog 
import tkMessageBox as mbox

import segmentation.make_selection

class MainForm(Frame):
  
    def __init__(self, parent):

        self.intput_folder = ''
        self.input_folder_tk = StringVar()        

        self.output_folder = ''
        self.output_folder_tk = StringVar()

        self.settings_filename = ''
        self.settings_filename_tk = StringVar()
        
        self.output_generate_graph_images = False
        self.output_generate_graph_images_tk = IntVar()

        self.output_generate_debug_images = False
        self.output_generate_debug_images_tk = IntVar()

        self.output_generate_final_mask = False
        self.output_generate_final_mask_tk = IntVar()

        self.output_generate_coordinate_file = False
        self.output_generate_coordinate_file_tk = IntVar()

        self.output_generate_metamorph_file = False
        self.output_generate_metamorph_file_tk = IntVar()
        
        self.param_clustersize_tk = StringVar()   
        self.param_clustersize_tk.set('1')      
        self.param_dist_tk = StringVar()
        self.param_dist_tk.set('1')
        self.param_random_selection_tk = IntVar()

        Frame.__init__(self, parent)            
        self.parent = parent
        self.initUI()
        
        
    def onOpenInput(self):
              
        #dlg = tkFileDialog.askdirectory(self, title="Open Input Folder", mustexist=True)
        self.input_folder = tkFileDialog.askdirectory()                
        self.input_folder_tk.set(self.input_folder)  
        self.update()      
        return     

    def onOpenSettings(self):
                      
        options = {'title': 'Choose a settings file', 
                   'filetypes': [('all files', '.*'), ('python files', '.py')]}
        self.settings_filename = tkFileDialog.askopenfilename(**options)              
        self.settings_filename_tk.set(self.settings_filename)  
        self.update()      
        return     

    def onOpenOutput(self):
              
        #dlg = tkFileDialog.askdirectory(self, title="Open Input Folder", mustexist=True)
        self.output_folder = tkFileDialog.askdirectory()                
        self.output_folder_tk.set(self.output_folder)  
        self.update()      
        return     


    def onStart(self):
        gui_settings = self.get_settings()
        if gui_settings is None:
            mbox.showwarning('Warning', 'Some value was not correctly entered. Processing was not started.')
            return

        ana = make_selection.Analyzer(gui_settings['settings_filename'])
        ana.process_folder(gui_settings['input_folder'], gui_settings['output_folder'])
        
        return
    
    def get_settings(self):
        self.settings = {}

        # read out all the settings and start the scripts.         
        self.input_folder = self.input_folder_tk.get()
        if not os.path.isdir(self.input_folder): 
            mbox.showerror("Error", "Input Folder does not exist.")
            return None
        self.settings['input_folder'] = self.input_folder
                
        self.output_folder = self.output_folder_tk.get()
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        self.settings['output_folder'] = self.output_folder
        
        self.settings_filename = self.settings_filename_tk.get()
        if not os.path.isfile(self.settings_filename):
            mbox.showerror("Error", "Settings file not found.")
            return None
        self.settings['settings_filename'] = self.settings_filename
        
        try:    
            cluster_size = int(self.param_clustersize_tk.get())
        except:
            mbox.showerror("Error", "Value for cluster size is not valid. Please enter an integer.")
            return None
        self.settings['cluster_size'] = cluster_size
        
        try:    
            cluster_dist = int(self.param_dist_tk.get())
        except:
            mbox.showerror("Error", "Value for distance between clusters is not valid. Please enter an integer.")
            return None
        self.settings['cluster_dist'] = cluster_dist

        self.settings['random_cell_selection'] = self.param_random_selection_tk.get() == 1

        self.settings['make_coordinate_file'] = self.output_generate_coordinate_file_tk.get() == 1
        self.settings['make_metamorph_file'] = self.output_generate_metamorph_file_tk.get() == 1
        
        self.settings['make_debug_images'] = self.output_generate_debug_images_tk.get() == 1
        self.settings['make_overlay_images'] = self.output_generate_graph_images_tk.get() == 1
        self.settings['make_final_image'] = self.output_generate_final_mask_tk.get() == 1
        
        return self.settings
    
    def initUI(self):

        self.parent.title("Cell Grid Illumination")

        st = Style()
        st.theme_use('aqua')
        self.parent.configure(bg='#ececec')
        master = self.parent

        input_settings = LabelFrame(self.parent, text="Input Settings")       
        parameter_settings = LabelFrame(self.parent, text="Parameter Settings")        
        output_settings = LabelFrame(self.parent, text="Output Settings")
        down_buttons = LabelFrame(self.parent, text="")
        
        ALLPAD = 8
        input_settings.grid(row=0, columnspan=3, sticky='WE', 
                            padx=ALLPAD, pady=ALLPAD, ipadx=ALLPAD, ipady=ALLPAD)
        parameter_settings.grid(row=3, columnspan=3, sticky='WE', 
                                padx=ALLPAD, pady=ALLPAD, ipadx=ALLPAD, ipady=ALLPAD)
        output_settings.grid(row=6, columnspan=3, sticky='WE', 
                             padx=ALLPAD, pady=ALLPAD, ipadx=ALLPAD, ipady=ALLPAD)
        down_buttons.grid(row=11, columnspan=2, sticky='WE', 
                          padx=ALLPAD, pady=ALLPAD, ipadx=ALLPAD, ipady=ALLPAD)

        Label(input_settings, text="Input Folder").grid(row=0, sticky=W)
        Label(input_settings, text="Output Folder").grid(row=1, sticky=W)
        Label(input_settings, text="Settings File").grid(row=2, sticky=W)
        
        self.e1 = Entry(input_settings, text=self.input_folder_tk, width=60) 
        self.e2 = Entry(input_settings, text=self.output_folder_tk, width=60) 
        self.e3 = Entry(input_settings, text=self.settings_filename_tk, width=60)
         
        self.open_button_in = Button(input_settings, text='Choose Folder ...', 
                                     command=self.onOpenInput)
        self.open_button_out = Button(input_settings, text='Choose Folder ...', 
                                      command=self.onOpenOutput)
        self.open_button_settings = Button(input_settings, text='Choose File ...', 
                                           command=self.onOpenSettings)
               
        self.e1.grid(row=0, column=1, sticky=W)        
        self.e2.grid(row=1, column=1, sticky=W)
        self.e3.grid(row=2, column=1, sticky=W)
        
        self.open_button_in.grid(row=0, column=2, sticky=W, pady=4)
        self.open_button_out.grid(row=1, column=2, sticky=W, pady=4)
        self.open_button_settings.grid(row=2, column=2, sticky=W, pady=4)

        # parameters
        Label(parameter_settings, text="Cluster size").grid(row=3, sticky=W)
        Label(parameter_settings, text="Cluster distance").grid(row=4, sticky=W)

        self.e4 = Entry(parameter_settings, text=self.param_clustersize_tk) 
        self.e5 = Entry(parameter_settings, text=self.param_dist_tk) 
        self.cb_random_selection = Checkbutton(output_settings, text="Random Cell Selection", variable=self.param_random_selection_tk)

        self.e4.grid(row=3, column=1, sticky=W)
        self.e5.grid(row=4, column=1, sticky=W)
        self.cb_random_selection.grid(row=5, column=0, sticky=W)
                
        # output        
        self.cb_output_generate_graph_images = Checkbutton(output_settings, text="Generate Graph Images", variable=self.output_generate_graph_images_tk)
        self.cb_output_generate_graph_images.grid(row=6, column=0, sticky=W, pady=4)

        self.cb_output_generate_debug_images = Checkbutton(output_settings, text="Generate All Intermediate Images", variable=self.output_generate_debug_images_tk)
        self.cb_output_generate_debug_images.grid(row=7, column=0, sticky=W, pady=4)

        self.cb_output_generate_final_mask = Checkbutton(output_settings, text="Generate Final Mask (single image)", variable=self.output_generate_final_mask_tk)
        self.cb_output_generate_final_mask.grid(row=8, column=0, sticky=W, pady=4)

        self.cb_output_generate_coordinate_file = Checkbutton(output_settings, text="Generate Coordinate Text ile", variable=self.output_generate_coordinate_file_tk)
        self.cb_output_generate_coordinate_file.grid(row=9, column=0, sticky=W, pady=4)

        self.cb_output_generate_metamorph_file = Checkbutton(output_settings, text="Generate Metamorph File", variable=self.output_generate_metamorph_file_tk)
        self.cb_output_generate_metamorph_file.grid(row=10, column=0, sticky=W, pady=4)
        
        Button(down_buttons, text='Start', command=self.onStart).grid(row=11, column=0, sticky=W, pady=4, padx=4)
        Button(down_buttons, text='Close', command=master.quit).grid(row=11, column=1, sticky=W, pady=4, padx=4)
        
        self.update()
        
        return
    
    
def main():
  
    root = Tk()
    root.geometry("880x600+300+300")
    #root.wm_iconbitmap(os.path.join('..','..','resources','Grill.ico'))
    app = MainForm(root)
    root.mainloop()  


if __name__ == '__main__':
    main()  
    
    