path_to_data = '/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/';

file = List2Cell(fullfile(path_to_data,'datasets_AdnPostSub2.list')); 

x = [];
y = [];

for i = 1:size(file)
    dset = file(i)           
    data_dir = fullfile(path_to_data, char(dset));
    cd(data_dir);
    binSize = 0.010; %in seconds
    load('Analysis/BehavEpochs.mat','wakeEp');
    load('Analysis/SpikeData.mat', 'S', 'shank');
    load('Analysis/HDCells.mat'); 
    load('Analysis/GeneralInfo.mat', 'shankStructure'); 
    [~,fbasename,~] = fileparts(pwd);
    [X,Y,~,wstruct] = LoadPosition_Wrapper(fbasename);
    [ang,angGoodEp] = HeadDirection_Wrapper(fbasename,wstruct);
    linSpd = LoadSpeed_Wrapper(fbasename,wstruct);
    hdC = hdCellStats(:,end)==1;    
    %Who were the hd cells recorded in the thalamus?
    thIx = hdC & ismember(shank,shankStructure{'thalamus'});
    %ordering their prefered direction
    [~,prefAngThIx] = sort(hdCellStats(thIx,1));
    %Who were the hd cells recorded in the postsub?
    poIx = hdC & ismember(shank,shankStructure{'postsub'});
    %ordering their prefered direction
    [~,prefAngPoIx] = sort(hdCellStats(poIx,1));
    %Restrict exploration to times were the head-direction was correctly
    %detected (you need to detect the blue and red leds, sometimes one of  the
    %two is just not visible)
    wakeEp  = intersect(wakeEp,angGoodEp);
    %Restrict all data to wake (i.e. exploration)
    S       = Restrict(S,wakeEp);
    ang     = Restrict(ang,wakeEp);
    X       = Restrict(X,wakeEp);
    Y       = Restrict(Y,wakeEp);
    linSpd  = Restrict(linSpd,wakeEp);
    %reinitialize indices (there may be hd cells that were not in the thalamus
    %nor in 0the postub. Well, actually it's not possible knowing the structure
    %of this dataset, but you never know)
    hdC     = thIx | poIx;
    thIx    = thIx(hdC);
    poIx    = poIx(hdC);
    %and restrict spike data to hd cells
    S = S(hdC);
    %Bin it!
    Q       = MakeQfromS(S,binSize);
    %And give some data
    dQ      = Data(Q);
    dQadn   = dQ(:,thIx);
    dQpos   = dQ(:,poIx);
    smWd = 2.^(0:8);
    dQadn   = gaussFilt(dQadn,5,0); 
    dQpos   = gaussFilt(dQpos,5,0);
    tStart  = Start(wakeEp,'min');
    tEnd    = End(wakeEp,'min');
    xlimEp  = [tStart(1) tEnd(end)];
    %Note to regress spike Data to position, you need to get the same timestamps for the two measures. Easy:
    Xq = Restrict(X,Q);
    Yq = Restrict(Y,Q);
    Aq = Restrict(ang,Q);
    Sp = Restrict(linSpd,Q);
    cd(path_to_data);    
    data_to_save = struct('X',  Data(Xq), 'Y',  Data(Yq), 'Ang', Data(Aq), 'speed', Data(Sp), 'ADn', dQadn(:,prefAngThIx), 'Pos', dQpos(:,prefAngPoIx));
    
    [m,n] = size(dQadn);
    x = [x n];    
    [m,n] = size(dQpos);
    y = [y n];
    
    tmp = strsplit(char(dset), '/');    
    save(strcat('/home/guillaume/Prediction_ML_GLM/python/data/sessions/wake/boosted_tree.', char(tmp(2)), '.mat'), '-struct', 'data_to_save');
end
    