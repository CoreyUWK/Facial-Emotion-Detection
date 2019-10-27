clear;
close all;
clc;
format compact;

imagesDirPath = '../cohn-kanade-images';
subjects = dir(imagesDirPath);
% folders = {'000' '001' '002' '003' '004' '005' '006' '007'};
emotionLabels = {'neutral' 'anger' 'contempt' 'disgust' 'fear' 'happy' 'sadness' 'surprise'};

allLandmarks = [];
emotionLabels = [];

% First two dirs are '.' and '..'
for subject = 1 : length(subjects)
    subjectName = subjects(subject).name
    subjectDirPath = strcat(imagesDirPath, '/', subjectName);
    subjectFolders = dir(subjectDirPath);
    
    if isdir(subjectDirPath) && ~strcmp(subjects(subject).name, '.') && ~strcmp(subjects(subject).name, '..')
        for folder = 1 : length(subjectFolders)
            imageDirPath = strcat(subjectDirPath, '/', subjectFolders(folder).name);
            subjectFolderName = subjectFolders(folder).name;
            if isdir(imageDirPath) && ~strcmp(subjectFolderName, '.') && ~strcmp(subjectFolderName, '..')
                imageDir = dir(imageDirPath);   % actual images

                if ~isempty(imageDir)
                    bestImageFile = imageDir(end).name;
                    [pathstr,imageName,ext] = fileparts(bestImageFile);

                    % Check if emotion is labelled
                    emotionDirPath = strcat('../Emotion/', subjectName, '/', subjectFolderName);
                    emotionFile = strcat(emotionDirPath, '/', imageName, '_emotion.txt');
                    if exist(emotionFile, 'file')
                        % zero indexed emotions
                        emotion = importdata(emotionFile) + 1;

                        if emotion == 2 || emotion == 6 || emotion == 7 || emotion == 8
                            % Add a column to the landmarks
                            landmarksDirPath = strcat('../Landmarks/', subjectName, '/', subjectFolderName);
                            landmarksFile = strcat(landmarksDirPath, '/', imageName, '_landmarks.txt');

                            landmarks = importdata(landmarksFile);
                            allLandmarks = [allLandmarks [landmarks(:,1); landmarks(:,2)]];

                            % Happy, Sad, Anger, Surprise
                            if emotion == 2
                                emotionLabel = 3;
                            elseif emotion == 6
                                emotionLabel = 1;
                            elseif emotion == 7
                                emotionLabel = 2;
                            else
                                emotionLabel = 4;
                            end

                            emotionLabels = [emotionLabels emotionLabel];
                        end
                    else
                        emotion = 0;
                    end
                end
            end
        end
    end
end

outputEmotions = full(ind2vec(emotionLabels));
save('Dataset.mat', 'allLandmarks', 'emotionLabels');