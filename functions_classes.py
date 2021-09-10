import numpy as np
import matplotlib.pyplot as plt



class evs_from_struct:
    def __init__(self, mat, LR): # left is 0, right is 1
        self.xs = mat['events'][0][0][LR][0][0][0]
        self.ys = mat['events'][0][0][LR][0][0][1]
        self.polarity = mat['events'][0][0][LR][0][0][2]
        self.timestep = mat['events'][0][0][LR][0][0][3].flatten()



def get_neighborhood(x,y,w,mats):
#     print(np.shape(x)[0], np.shape(y)[0], np.shape(w), np.shape(mats))
    r = (w-1)/2
    TSs = np.zeros((w, w*np.shape(mats)[2], np.shape(x)[0]))
    
    # iterate through the events
    for ti in range(np.shape(x)[0]):
        # preallocate space for time surface
        temp_TS = np.zeros((w,w*np.shape(mats)[2]))
        # iterate through Time windows
        for tj in range(np.shape(mats)[2]):
            temp_TS[:,(tj+1)*w-w:(tj+1)*w] = mats[int(x[ti]-r-1):int(x[ti]+r), int(y[ti]-r-1):int(y[ti]+r), tj]
        
        TSs[:,:,ti] = temp_TS
        
    return TSs



def downscale(TSs, kernel_size):
      
    dim0 = np.ceil(np.shape(TSs)[0]/kernel_size[0])
    dim1 = np.ceil(np.shape(TSs)[1]/kernel_size[1])

    TSs_size0 = np.shape(TSs)[0]
    TSs_size1 = np.shape(TSs)[1]

#     print(dim0, dim1)
#     print('TSs shape', np.shape(TSs))
    TSs_scaled = np.zeros((np.shape(TSs)[2], int(dim0*dim1)))
#     print('TSs_scaled', np.shape(TSs_scaled))

    for i in range(np.shape(TSs)[2]):
#         print(np.shape(TSs))
#         temp = TSs[0:kernel_size[0]:-1, 0:kernel_size[1]:-1, i]
        temp = TSs[::kernel_size[0], ::kernel_size[1], i]
#         plt.imshow(temp)
#         plt.pause(.2)
#         plt.imshow(TSs[:,:,i])
#         plt.pause(.2)
#         print('hello')
#         print('subsampled shape', np.shape(temp))
        TSs_scaled[i,:] = temp.flatten()
#         print('flattened shape', np.shape(temp.flatten()))
        
    return TSs_scaled



def dist_measure(l_feat, r_feat):

    # resulting matrix is left events as rows and right events as columns
    D = np.matmul(l_feat, r_feat.T)

    return D


def disparity(winner_pairs, l_xs, l_ys, r_xs, r_ys, temp_map):

    C = 25  
    f = 10  
    b = 10
    dim = np.shape(temp_map)[0]
    
    winner_l_idx = [p[1] for p in winner_pairs]
    winner_r_idx = [p[0] for p in winner_pairs]

    dist_i = dim + r_xs[winner_r_idx] - l_xs[winner_l_idx]
    Z = f*b/dist_i
    X = (Z/f * l_xs[winner_l_idx] * C).astype(int)
    Y = (Z/f * l_ys[winner_l_idx] * C).astype(int)
    temp_map[X,Y] = Z
        
    return temp_map, X, Y, Z

