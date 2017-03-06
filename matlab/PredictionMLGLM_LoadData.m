dset = 'Mouse28/Mouse28-140313';

%should be define separately in the general workflow so that we can all use
%the same scripts, for example we could each have our own
%PredictionMLGLM_Init.m where this is defined.
%path_to_data = 'D:\Dropbox (Peyrache Lab)\Peyrache Lab Team Folder\Data\HDCellData';
path_to_data = '~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/';

data_dir = fullfile(path_to_data,dset);
cd(data_dir);

%Parameters
binSize = 0.1; %in seconds


%when the animal was exploring the arena
load('Analysis/BehavEpochs.mat','wakeEp');

%Spike Data
load('Analysis/SpikeData.mat', 'S', 'shank');

%All infor regarding HD cells
load('Analysis/HDCells.mat'); 

%needed to know on which electrode group each cell is recorded
load('Analysis/GeneralInfo.mat', 'shankStructure'); 

%load position
[~,fbasename,~] = fileparts(pwd);
[X,Y,~,wstruct] = LoadPosition_Wrapper(fbasename);

%Load head-direction (wstuct is the raw position data, saves some time not
%to relaod the text file)
[ang,angGoodEp] = HeadDirection_Wrapper(fbasename,wstruct);

%and speed
linSpd = LoadSpeed_Wrapper(fbasename,wstruct);

%boolean index of who are HD cells
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

dQadn   = gaussFilt(dQadn,5,0);
dQpos   = gaussFilt(dQpos,5,0);

tStart  = Start(wakeEp,'min');
tEnd    = End(wakeEp,'min');
xlimEp  = [tStart(1) tEnd(end)];

figure(1),clf
set(gcf,'Position',[100 145 1618 797])
subplot(6,1,1)
    imagesc(Range(Q,'min'),(1:sum(thIx)),dQadn(:,prefAngThIx)');
    xlim(xlimEp);
    ylabel('ADn cells')
subplot(6,1,2)
    imagesc(Range(Q,'min'),(1:sum(poIx)),dQpos(:,prefAngPoIx)');
    xlim(xlimEp);
    ylabel('PoS cells')
subplot(6,1,3)
    plot(Range(X,'min'),Data(X));
    xlim(xlimEp);
    ylabel('X pos (cm)')
subplot(6,1,4)
    plot(Range(Y,'min'),Data(Y));
    xlim(xlimEp);
    ylabel('Y pos (cm)')
subplot(6,1,5)
    plot(Range(ang,'min'),Data(ang));
    xlim(xlimEp);
    ylabel('HD (rad)')
subplot(6,1,6)
    plot(Range(linSpd,'min'),Data(linSpd));
    xlim(xlimEp);
    ylabel('speed (cm/s)')
    
%Note to regress spike Data to position, you need to get the same timestamps for the two measures. Easy:
Xq = Restrict(X,Q);
Yq = Restrict(Y,Q);
Aq = Restrict(ang,Q);
Sp = Restrict(linSpd,Q);
cd(path_to_data);    

data_to_save = struct('X',  Data(Xq), 'Y',  Data(Yq), 'Ang', Data(Aq), 'speed', Data(Sp), 'ADn', dQadn(:,prefAngThIx), 'Pos', dQpos(:,prefAngPoIx));
%save('data_test_boosted_tree.mat', '-struct', 'data_to_save');
