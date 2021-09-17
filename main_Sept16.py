import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import cv2
import os
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation

##########################################################################

def get_neighborhood(x, y, mats, w, kernel_size):

    # define the neighborhood dimension
    r = (w-1)/2

    # preallocate space for time surfaces (rows = num events, cols = num pixels in resized time surface)
    TSs = np.zeros((np.shape(x)[0], np.shape(mats)[0]*int(w/kernel_size[0]+1)*int(w/kernel_size[1]+1)))
    
    # iterate through the events
    for ti in range(np.shape(x)[0]):
        
        # get the patches around event ti
        temp_TS = mats[:, int(x[ti]-r-1):int(x[ti]+r), int(y[ti]-r-1):int(y[ti]+r)]
        
        # resize the patches
        resized_TS = temp_TS[:, ::kernel_size[0], ::kernel_size[1]]
        
        # flatten the resized patches and store them
        TSs[ti, :] = resized_TS.reshape(1,np.shape(resized_TS)[0]*np.shape(resized_TS)[1]*np.shape(resized_TS)[2])
       
    return TSs


def disparity(winner_pairs, l_xs, l_ys, r_xs, r_ys, temp_map):

    # define parameters for triangulation
    C = 25  
    f = 10  
    b = 10
    dim = np.shape(temp_map)[0]
    
    # get the indices of the matching events in the left and right frames
    winner_l_idx = [p[1] for p in winner_pairs]
    winner_r_idx = [p[0] for p in winner_pairs]

    # calculate disparity
    dist_i = dim + r_xs[winner_r_idx] - l_xs[winner_l_idx]
    
    # calculate depth and true coordiantes
    Z = f*b/dist_i
    X = (Z/f * l_xs[winner_l_idx] * C).astype(int)
    Y = (Z/f * l_ys[winner_l_idx] * C).astype(int)
    temp_map[X,Y] = Z
        
    return temp_map, X, Y, Z


class evs_from_struct:
    def __init__(self, mat, LR): # left is 0, right is 1
        self.xs = mat['events'][0][0][LR][0][0][0]
        self.ys = mat['events'][0][0][LR][0][0][1]
        self.polarity = mat['events'][0][0][LR][0][0][2]
        self.timestep = mat['events'][0][0][LR][0][0][3].flatten()


##########################################################################


def main():

    mat = io.loadmat('/Users/nolanardolino/Desktop/UCSD_research/Stereo_qDVS/matlab_semidense/DVS Stereo dataset/StereoEventDataset/fan_distance1_orientation1_DAVIS.mat');

    L = evs_from_struct(mat,0) # left camera data
    R = evs_from_struct(mat,1) # right camera data

    POI = 1 # polarity of interest
    L_idxs_pol = [idx for idx in range(len(L.polarity)) if L.polarity[idx] == POI]
    R_idxs_pol = [idx for idx in range(len(R.polarity)) if R.polarity[idx] == POI]

    L.xs = np.array([L.xs[i] for i in L_idxs_pol])
    L.ys = np.array([L.ys[i] for i in L_idxs_pol])
    L.polarity = np.array([L.polarity[i] for i in L_idxs_pol])
    L.timestep = np.array([L.timestep[i] for i in L_idxs_pol])

    R.xs = np.array([R.xs[i] for i in R_idxs_pol])
    R.ys = np.array([R.ys[i] for i in R_idxs_pol])
    R.polarity = np.array([R.polarity[i] for i in R_idxs_pol])
    R.timestep = np.array([R.timestep[i] for i in R_idxs_pol])

    num_ticks = min(max(L.timestep), max(R.timestep))
    num_L_events = len(L.timestep)
    num_R_events = len(R.timestep)

    
    w = 51
    dim = 200
    T_windows = [1, 5, 10]
    kernel_size = [2,2]

    L_frames = []
    R_frames = []

    # turn events into frames
    for t in range(num_ticks):
        L_frame = np.zeros((dim,dim))
        R_frame = np.zeros((dim,dim))
        L_frame[L.xs[L.timestep == t], L.ys[L.timestep == t]] = 1
        R_frame[R.xs[R.timestep == t], R.ys[R.timestep == t]] = 1
        
        L_frames.append(np.pad(L_frame>0, w, mode='constant', constant_values=False))
        R_frames.append(np.pad(R_frame>0, w, mode='constant', constant_values=False))

    frame_dim = np.shape(L_frames[0])


    depth_frames = []

    num_ticks = 100
    for i in range(0,num_ticks):

        # define which frames should be selected to accommodate the given time windows
        frame_idxs = [np.arange(max(0,i-tw),max(i,1)) for tw in T_windows]
        L_tws = np.zeros((len(T_windows), frame_dim[0], frame_dim[1]))
        R_tws = np.zeros((len(T_windows), frame_dim[0], frame_dim[1]))
        
        # iterate through time windows and collect events
        for tw in range(len(T_windows)):
            L_sum_tw = np.sum(L_frames[frame_idxs[tw][0]:(frame_idxs[tw][-1]+1)], axis=0)>0
            R_sum_tw = np.sum(R_frames[frame_idxs[tw][0]:(frame_idxs[tw][-1]+1)], axis=0)>0

            L_tws[tw,:,:] = L_sum_tw
            R_tws[tw,:,:] = R_sum_tw      
        
        # find the location of 1s in T_windows[0] = 1
        L_tw1_x, L_tw1_y = np.where(L_tws[0,:,:] == 1)
        R_tw1_x, R_tw1_y = np.where(R_tws[0,:,:] == 1)
        
        
        # collect event patches from each time window corresponding to events in tw1, resize, and flatten
        L_TSs_flat = get_neighborhood(L_tw1_x, L_tw1_y, L_tws, w, kernel_size)
        R_TSs_flat = get_neighborhood(R_tw1_x, R_tw1_y, R_tws, w, kernel_size)
            
        # pairwise comparison
        Dlr = np.matmul(L_TSs_flat, R_TSs_flat.T)

        # lr -> left events as rows / right events as columns
        # rl -> opposite
        lr_best_matches = [np.arange(np.shape(Dlr)[0])[Dlr[:,i] == np.max(Dlr[:,i])] for i in range(np.shape(Dlr)[1])]
        rl_best_matches = [np.arange(np.shape(Dlr)[1])[Dlr[i,:] == np.max(Dlr[i,:])] for i in range(np.shape(Dlr)[0])]
            
        winner_pairs = [] # left, right pairs
        # iterate through left camera pixels
        for l_pix, r_pix_match in enumerate(lr_best_matches):
            # iterate through pixels that l_pix matched with
            for r_pix in r_pix_match: # these are indices
                # see if the match is reciprocated
                if l_pix in rl_best_matches[r_pix]:
                    winner_pairs.append([l_pix, r_pix])
                
        temp_map = np.zeros((np.shape(L_tws)[1],np.shape(L_tws)[2]))
        depth_map, X, Y, Z = disparity(winner_pairs, L_tw1_x, L_tw1_y, R_tw1_x, R_tw1_y, temp_map)
        depth_frames.append(depth_map)
            # every 5 ticks, do a sanity check
        if i%5 == 0:
            print(i, 'out of', num_ticks)
            plt.imshow(depth_map)
            plt.pause(.2)
        

    def animate(frame):
        f = depth_frames[frame]
        line.set_data(f)

    fig = plt.figure(10)    
    line = plt.imshow([[]], extent=(0,10,0,10), cmap='viridis', clim=(0,1))
    anim = animation.FuncAnimation(fig, animate, frames=num_ticks, interval=60) 
    HTML(anim.to_jshtml())   

    writervideo = animation.FFMpegWriter(fps=60)
    anim.save('fan_depth_video.mp4', writer=writervideo)
        
if __name__ == '__main__':
    main()