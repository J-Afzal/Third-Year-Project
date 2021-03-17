%% Junaid Afzal
%% Load in data
clear variables;
close all;

% Platforms
windows = 'Windows 10 Desktop';
linux = 'Linux (Ubuntu 20.04) Desktop';
jetson = 'Jetson Nano (4GB)';

% File arrays
windowsFiles = dir(strcat(windows, '\*.txt'));
linuxFiles = dir(strcat(linux, '\*.txt'));
jetsonFiles = dir(strcat(jetson, '\*.txt'));

% Consts
numberOfFiles = length(windowsFiles);
numberOfDataPoints = 1155;
numberOfTests = 11;
videoLocation = '..\- MATLAB Videos\';
FPS = 30;
x = 1:1:numberOfDataPoints;

% Read in files
windowsFilesData = cell(numberOfTests, 1);
linuxFilesData = cell(numberOfTests, 1);

windowsFilesData{1} = importdata(strcat(windowsFiles(1).folder, '\', windowsFiles(1).name));
linuxFilesData{1} = importdata(strcat(linuxFiles(1).folder, '\', linuxFiles(1).name));
testFiles(1) = windowsFiles(1);

j=2;
for i=3:2:numberOfFiles
    windowsFilesData{j} = importdata(strcat(windowsFiles(i).folder, '\', windowsFiles(i).name));
    linuxFilesData{j} = importdata(strcat(linuxFiles(i).folder, '\', linuxFiles(i).name));
    testFiles(j) = windowsFiles(i);
    j = j + 1;
end

jetsonFilesData = cell(numberOfTests, 1);
for i=1:numberOfTests
    jetsonFilesData{i} = importdata(strcat(jetsonFiles(i).folder, '\', jetsonFiles(i).name));
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
    title(strcat('Frame times for ', testFiles(i).name), 'Interpreter', 'none');
    legend(' Windows 10 Desktop', ' Linux (Ubuntu 20.04) Desktop', ' Jetson Nano (4GB)', 'location','southoutside');

    outputVideo = VideoWriter(strcat(testFiles(i).folder, '\', videoLocation, testFiles(i).name), 'MPEG-4');
    outputVideo.FrameRate = FPS;    
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