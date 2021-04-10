%% Junaid Afzal
%% Load in data
clear variables;
close all;

% Platforms
windows = 'Output for Windows 10 Desktop';
linux = 'Output for Linux Desktop';
jetson = 'Output for Jetson Nano';

% File arrays
windowsFiles = dir(strcat(windows, '/'));
linuxFiles = dir(strcat(linux, '/'));
jetsonFiles = dir(strcat(jetson, '/'));

% Consts
numberOfFiles = length(windowsFiles);
numberOfDataPoints = 1155; % Ignore first frame due to being very high
numberOfTests = 11;
videoLocation = 'Animated Graphs\';
videoFPS = 30;
x = 1:1:numberOfDataPoints;

% Read in files
windowsFilesData = cell(numberOfTests, 1);
linuxFilesData = cell(numberOfTests, 1);

windowsFilesData{1} = importdata(strcat(windowsFiles(3).folder, '/', windowsFiles(3).name));
linuxFilesData{1} = importdata(strcat(linuxFiles(3).folder, '/', linuxFiles(3).name));
testFiles(1) = windowsFiles(3); % to extract name info

j=2;
for i=4:2:numberOfFiles
    windowsFilesData{j} = importdata(strcat(windowsFiles(i).folder, '/', windowsFiles(i).name));
    linuxFilesData{j} = importdata(strcat(linuxFiles(i).folder, '/', linuxFiles(i).name));
    testFiles(j) = windowsFiles(i);
    j = j + 1;
end

jetsonFilesData = cell(numberOfTests, 1);
j = 1;
for i=3:numberOfTests+2
    jetsonFilesData{j} = importdata(strcat(jetsonFiles(i).folder, '/', jetsonFiles(i).name));
    j = j + 1;
end

%% Frame Times
for i=1:numberOfTests
    figure1 = figure;
    set(gcf, 'Position', [100, 100, 850, 700]);

    % Animated Lines
    windowsLine = animatedline('Color', [0.25 0.5 1], 'LineWidth', 1);
    linuxLine = animatedline('Color', [0.5 0.5 0.5], 'LineWidth', 1);
    jetsonLine = animatedline('Color', [0.5 0.9 0.25], 'LineWidth', 1);

    % Graph props
    xlabel('Frame Number');
    ylabel('Time to compute frame (ms)');
    xlim([0 1155]);
    grid on;
    title(strcat('Frame times for ', {' '}, testFiles(i).name), 'Interpreter', 'none');
    legend(' Windows 10 Desktop', ' Linux Desktop', ' Jetson Nano', 'location','southoutside');

    outputVideo = VideoWriter(strcat(testFiles(i).folder, '\..\', videoLocation, testFiles(i).name), 'MPEG-4');
    outputVideo.FrameRate = videoFPS;    
    open(outputVideo);
    
    for j=1:numberOfDataPoints
        addpoints(windowsLine, x(j),windowsFilesData{i}(j));
        addpoints(linuxLine, x(j),linuxFilesData{i}(j));
        addpoints(jetsonLine, x(j),jetsonFilesData{i}(j));
        drawnow limitrate;        
        img = getframe(gcf);
        writeVideo(outputVideo,img)
    end
end

clear variables;
close all;