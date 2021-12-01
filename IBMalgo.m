


%% visualize DAVIS frames
close all
clear all

evs = load('events.mat');


% fan_data = load('/Users/nolanardolino/Desktop/UCSD_research/Stereo_qDVS/matlab_semidense/DVS Stereo dataset/StereoEventDataset/fan_distance1_orientation1_DAVIS.mat');
% S = fan_data;
% w = 25; % patch side length (odd number)
% kernel_size = [10,10];
w = 25; % patch side length (odd number)
kernel_size = [2,2];


dim = 300+2*w;

T_windows = [1, 5];

l_evs = zeros(dim,dim,size(T_windows,2));
r_times = zeros(dim,dim,size(T_windows,2));

for i = 1:1:max(evs.L.fs)

    % this is where events that belong to different time windows are stored
    % for the left and right frames
    l_evs = zeros(dim,dim,size(T_windows,2));
    r_evs = zeros(dim,dim,size(T_windows,2));
    
    % find all events that have already occured (given full list of events)
    l_temp_1 = evs.L.fs(evs.L.fs <= i);
    r_temp_1 = evs.R.fs(evs.R.fs <= i);

    % mark events that occur in each time window
    for t = 1:size(T_windows,2)

        l_temp = find(evs.L.fs <= i & evs.L.fs > max(0,i - T_windows(t)));
        r_temp = find(evs.R.fs <= i & evs.R.fs > max(0,i - T_windows(t)));
            
        l_xs = evs.L.xs(l_temp)+w;
        l_ys = evs.L.ys(l_temp)+w;
        
        r_xs = evs.R.xs(r_temp)+w;
        r_ys = evs.R.ys(r_temp)+w;

        l_evs_temp = zeros(dim,dim);
        r_evs_temp = zeros(dim,dim);
        
        l_evs_temp(sub2ind([dim,dim],l_xs,l_ys)) = 1;
        r_evs_temp(sub2ind([dim,dim],r_xs,r_ys)) = 1;
        
        l_evs(:,:,t) = l_evs_temp;
        r_evs(:,:,t) = r_evs_temp;
        
    end
    
    % now extract and concatenate time surfaces around events for T_window = 1
    l_temp = find(evs.L.fs <= i & evs.L.fs > max(0,i - 1));
    r_temp = find(evs.R.fs <= i & evs.R.fs > max(0,i - 1));

    l_xs = evs.L.xs(l_temp)+w;
    l_ys = evs.L.ys(l_temp)+w;

    r_xs = evs.R.xs(r_temp)+w;
    r_ys = evs.R.ys(r_temp)+w;
  
    l_evs_temp = zeros(dim,dim);
    r_evs_temp = zeros(dim,dim);

    l_evs_temp(sub2ind([dim,dim],l_xs,l_ys)) = 1;
    r_evs_temp(sub2ind([dim,dim],r_xs,r_ys)) = 1;
    
    TSs_l = get_neighborhood(l_xs,l_ys,w,l_evs);
    TSs_r = get_neighborhood(r_xs,r_ys,w,r_evs);
    
    % spatial scaling of the concatenated event patches from different time windows
    TSs_scaled_l = downscale(TSs_l, kernel_size);
    TSs_scaled_r = downscale(TSs_r, kernel_size);
    
    % L->R check
    % Compute the hadamard product between all left and right features
    Dlr = pdist2(TSs_scaled_l, TSs_scaled_r, @distfun); % dimensions is AxB | D matrix contain the hadamard product between all left and right timesurfaces
    
    pos1 = find(Dlr == max(Dlr,[],1));
    pos2 = find(Dlr == max(Dlr,[],2));
    
    matches = intersect(pos1, pos2);

    matchedx = [];
    matchedy = [];
    winner_pairs = {};
    for m = 1:numel(matches)
        [x,y] = ind2sub(size(Dlr), matches(m));
        if any(matchedx == x) || any(matchedy==y)
            continue
        end
        matchedx(end+1) = x;
        matchedy(end+1) = y;
        winner_pairs{end+1} = [x,y];
    end
    
%     disp(winner_pairs)

    temp_map = zeros(dim);
    temp_map_out = disparity(winner_pairs, l_xs, l_ys, r_xs, r_ys, temp_map);
    
    if rem(i,3) == 0


    end

    if rem(i,3) == 0
        figure(1);
        hold on
        imshow(imresize(temp_map_out, 3), [0, .001])
        colormap(jet) 
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function TSs = get_neighborhood(x,y,w,mats)

    r = (w-1)/2;
    TSs = zeros(w,w*size(mats,3),size(x,1));
    for ti = 1:size(x,1)
        temp_TS = zeros(w,w*size(mats,3));
        for tj = 1:size(mats,3)
            temp_TS(:,tj*w-w+1:tj*w) = mats(x(ti)-r:x(ti)+r, y(ti)-r:y(ti)+r, tj);
        end
        TSs(:,:,ti) = temp_TS;
    end
end

function TSs_scaled = downscale(TSs, kernel_size)
    
    dim1 = ceil(size(TSs,1)/kernel_size(1));
    dim2 = ceil(size(TSs,2)/kernel_size(2));
    
    TSs_size1 = size(TSs,1);
    TSs_size2 = size(TSs,2);
    
    TSs_scaled = zeros(size(TSs,3),dim1*dim2);
    for i = 1:size(TSs,3)
        temp = TSs(1:kernel_size(1):end,1:kernel_size(2):end,i);
        TSs_scaled(i,:) = temp(:);
    end

end

function D2 = distfun(ZI, ZJ)
    D2 = ZI * transpose(ZJ);
end


function temp_map = disparity(winner_pairs, l_xs, l_ys, r_xs, r_ys, temp_map)

% define parameters for triangulation
C = 45;
f = 10;
b = 10;
dim = size(temp_map);
% dim=0;

winner_l_idx = cellfun(@(v)v(1),winner_pairs);
winner_r_idx = cellfun(@(v)v(2),winner_pairs);

if numel(winner_l_idx) == 0 || numel(winner_r_idx) == 0
    return
end

dist_i = dim + r_xs(winner_r_idx) - l_xs(winner_l_idx);
% disp(dist_i)

Z = f*b./dist_i;
X = uint8(Z./f .* l_xs(winner_l_idx) * C);
Y = uint8(Z./f .* l_ys(winner_l_idx) * C);

I = sub2ind(size(temp_map),X,Y);

temp_map(I) = Z;
    
% disp(Z)

end
