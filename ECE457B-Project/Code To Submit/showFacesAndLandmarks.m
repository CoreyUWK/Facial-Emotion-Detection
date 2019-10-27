clear;
close all;
clc;

subjects = dir('../cohn-kanade-images/');
folders = {'000' '001' '002' '003' '004' '005' '006' '007'};
emotionLabels = {'neutral' 'anger' 'contempt' 'disgust' 'fear' 'happy' 'sadness' 'surprise'};

% First two dirs are '.' and '..'
for subject = 3 : length(subjects)
    foldername = subjects(subject).name;
    
    for folder = 1 : length(folders)
        % Images one folder up
        imageDirPath = strcat('../cohn-kanade-images/', foldername, '/', folders{folder});
        imageDir = dir(imageDirPath);
        
        if ~isempty(imageDir)
            bestImageFile = imageDir(end).name;
            [pathstr,imageName,ext] = fileparts(bestImageFile);
            bestImage = imread(strcat(imageDirPath, '/', bestImageFile));
            
            landmarksDirPath = strcat('../Landmarks/', foldername, '/', folders{folder});
            landmarksFile = strcat(landmarksDirPath, '/', imageName, '_landmarks.txt');
            landmarks = importdata(landmarksFile);
            
            emotionDirPath = strcat('../Emotion/', foldername, '/', folders{folder});
            emotionFile = strcat(emotionDirPath, '/', imageName, '_emotion.txt');
            if exist(emotionFile, 'file')
                % zero indexed emotions
                emotion = importdata(emotionFile) + 1;
            else
                emotion = 0;
            end
            
            [imHeight imWidth] = size(bestImage);
            
            figure(1);
            imshow(bestImage);
            hold on;
            scatter(landmarks(:,1),landmarks(:,2),'r.');
            for landmark = 1 : length(landmarks)
                label = num2str(landmark,'%d');
                text(landmarks(landmark,1)+0.15,landmarks(landmark,2)-0.15,label, ...
                    'horizontal','left','vertical','bottom','FontSize',8,'Color','white');
            end
            if emotion ~= 0
                title(emotionLabels{emotion});
            end
            
            [landmarks(:,1); landmarks(:,2)];
            hold off;
            pause
        end
    end
end
