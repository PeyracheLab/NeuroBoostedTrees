path_to_data = '/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HPC-NAc/';

all_subfolders = dir(path_to_data);
all_subfolders(1:2) = [];

for i = 1:size(all_subfolders)
    dset = all_subfolders(i).name
    
    data_dir = fullfile(path_to_data, dset);

    cd(data_dir);

    if exist(fullfile(data_dir, 'HPCCellClass_v1.mat'), 'file')
    
        load('NAcSpikes.mat');
        load('BehavEpochs.mat');
        load('HPCSpikes.mat');
        load('HPCCellClass_v1.mat');
        load('NAcCellClass_v1.mat');
        load('GeneralInfo_v1.mat');
        S = Restrict(NAcSpikes.S, wakeEp);
        Q = MakeQfromS(S, 0.025);
        Q = Data(Q);



        S = [Restrict(NAcSpikes.S,wakeEp);Restrict(HPCSpikes.S,wakeEp)];
        threshold = Rate(S) > 0.01;

        Q = MakeQfromS(S, 0.025);
        Q = Data(Q);

        Qnac = Q(:,1:length(NAcSpikes.S));
        Qhpc = Q(:,length(NAcSpikes.S)+1:end);
    
        tmp = Qhpc;
    
        index_hpc = threshold(length(NAcSpikes.S)+1:end)==1 & HPCCellClass_v1.isPYR==1;
        index_nac = threshold(1:length(NAcSpikes.S))==1 & NAcCellClass_v1.isMSN==1;

        Qhpc = Qhpc(:,index_hpc);
        Qnac = Qnac(:,index_nac);
    
        if ~isempty(Qhpc) && ~isempty(Qnac)
    
            fQnac = gaussFilt(Qnac, 10, 0);
            fQhpc = gaussFilt(Qhpc, 10, 0);
    
            data_to_save = struct('hpc', fQhpc, 'nac', fQnac);

            dset = split(dset, '_');

            filename = string('data_hipp_nac_'+dset(1)+'_'+GeneralInfo.sessionType+'.mat');

            save(filename.char, '-struct', 'data_to_save');
        end
    end
end
