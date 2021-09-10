import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from functions_classes import *



def main():

    mat = io.loadmat('/Users/nolanardolino/Desktop/UCSD_research/Stereo_qDVS/matlab_semidense/DVS Stereo dataset/StereoEventDataset/fan_distance1_orientation1_DAVIS.mat')

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


    w = 60
    kernel_size = [2,2]
    dim = 200 + 2*w

    T_windows = [2, 10, 50]

    L_ev_idxs = np.arange(len(L.timestep))
    R_ev_idxs = np.arange(len(R.timestep))

    data = []

    # iterate through each frame t
    for i in 1+np.arange(num_ticks):

        # allocate space for events to be drawn
        l_evs = np.zeros((dim,dim,np.shape(T_windows)[0]))
        r_evs = np.zeros((dim,dim,np.shape(T_windows)[0]))
        
        # identify which events have already occured
        L_past_events_idxs = L_ev_idxs[L.timestep <= i]
        R_past_events_idxs = R_ev_idxs[R.timestep <= i]
        
        # store events in their allocated spaces according to the time window
        for t in range(np.shape(T_windows)[0]):

            # find which events have occured in time window t
            l_temp = L_past_events_idxs[L.timestep[L_past_events_idxs] > max(0, i - T_windows[t])]
            r_temp = R_past_events_idxs[R.timestep[R_past_events_idxs] > max(0, i - T_windows[t])]

            l_xs = L.xs[l_temp].flatten() + w
            l_ys = L.ys[l_temp].flatten() + w
            
            r_xs = R.xs[r_temp].flatten() + w
            r_ys = R.ys[r_temp].flatten() + w
            
            l_evs_temp = np.zeros((dim,dim))
            r_evs_temp = np.zeros((dim,dim))
            
            l_evs_temp[l_xs, l_ys] = 1
            r_evs_temp[r_xs, r_ys] = 1
            
            l_evs[:,:,t] = l_evs_temp
            r_evs[:,:,t] = r_evs_temp
            
        # find all events that have taken place in the last timestep to get time surfaces
        l_temp = L_past_events_idxs[L.timestep[L_past_events_idxs] > max(0, i - T_windows[t])]
        r_temp = R_past_events_idxs[R.timestep[R_past_events_idxs] > max(0, i - T_windows[t])]

        l_xs = L.xs[l_temp].flatten() + w
        l_ys = L.ys[l_temp].flatten() + w

        r_xs = R.xs[r_temp].flatten() + w
        r_ys = R.ys[r_temp].flatten() + w
            
        # extract and contatentate time surfaces from different time windows
        TSs_l = get_neighborhood(l_xs, l_ys, w, l_evs)
        TSs_r = get_neighborhood(r_xs, r_ys, w, r_evs)
            

        # spatial scaling of the concatenated event patches from different time windows
        TSs_scaled_l = downscale(TSs_l, kernel_size)
        TSs_scaled_r = downscale(TSs_r, kernel_size)

        
        # pairwise comparison
        Dlr = dist_measure(TSs_scaled_l, TSs_scaled_r)
        
        
        # lr -> left events as rows / right events as columns
        # rl -> opposite
        lr_best_matches = [np.arange(np.shape(Dlr)[0])[Dlr[:,i] == np.max(Dlr[:,i])] for i in range(np.shape(Dlr)[1])]
        rl_best_matches = [np.arange(np.shape(Dlr)[1])[Dlr[i,:] == np.max(Dlr[i,:])] for i in range(np.shape(Dlr)[0])]
            
        winner_pairs = [] # left, right pairs
        for l_pix, r_pix_match in enumerate(lr_best_matches):
            
            for r_pix in r_pix_match: # these are indices
                
                # see if the match is reciprocated
                if l_pix in rl_best_matches[r_pix]:
                    winner_pairs.append([l_pix, r_pix])
                
        temp_map = np.zeros((dim,dim))
        disparity_map, X, Y, Z = disparity(winner_pairs, l_xs, l_ys, r_xs, r_ys, temp_map)
        data.append([X,Y,Z])
        
            # every 5 ticks, do a sanity check
        if i%5 == 0:
            print(i, 'out of', num_ticks)
            # plt.imshow(TSs_scaled_l)
            # plt.pause(.2)
            # plt.imshow(TSs_l[:,:,0])
            # plt.pause(.2)
            plt.imshow(disparity_map)
            plt.pause(.2)
        
        



if __name__ == "__main__":
    main()