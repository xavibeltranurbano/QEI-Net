% Directory where the folders are stored
rootDir = '/Users/xavibeltranurbano/Desktop/UPenn/Standard_Dataset/Datasets/QEI-Dataset';

% Get a list of all subdirectories
dirs = dir(rootDir);
isub = [dirs(:).isdir]; % returns logical vector
nameFolds = {dirs(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];

% Loop over each directory
for i = 1:length(nameFolds)
    subdir = fullfile(rootDir, nameFolds{i});
    files = dir(fullfile(subdir, '*CBF_Map.nii')); % Adjust the pattern to match your file names
    for j = 1:length(files)
        filename = files(j).name;
        fullfileInput = fullfile(subdir, filename);
        % Define the smoothed file names
        spmSmoothedFile = fullfile(subdir, 'CBF_Map_smoothed.nii');
        % SPM smoothing
        spm_smooth(fullfileInput, spmSmoothedFile, [5 5 5]);
    end
end
