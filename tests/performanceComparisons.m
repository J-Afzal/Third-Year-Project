%% Junaid Afzal
%% Load in data
clear variables;
close all;

% Platforms
windows = 'Windows 10 Desktop';
linux = 'Linux (Ubuntu 20.04) Desktop';
jetson = 'Jetson Nano (4GB)';

% File arrays
windowsFiles = dir(strcat(windows, '/*.txt'));
linuxFiles = dir(strcat(linux, '/*.txt'));
jetsonFiles = dir(strcat(jetson, '/*.txt'));

% Consts
numberOfFiles = length(windowsFiles);
numberOfDataPoints = 1155;
numberOfTests = 11;
x = 1:1:numberOfDataPoints;

% Read in files
windowsFilesData = cell(numberOfFiles, 1);
linuxFilesData = cell(numberOfFiles, 1);
for i=1:numberOfFiles
    windowsFilesData{i} = importdata(strcat(windowsFiles(i).folder, '\', windowsFiles(i).name));
    linuxFilesData{i} = importdata(strcat(linuxFiles(i).folder, '\', linuxFiles(i).name));
end

% Calculate the average FPS for each file
windowsAverageFPS = zeros(numberOfFiles, 1);
linuxAverageFPS = zeros(numberOfFiles, 1);
for i=1:numberOfFiles
    windowsTotal = 0;
    linuxTotal = 0;
    for j=2:numberOfDataPoints-1
        windowsTotal = windowsTotal + windowsFilesData{i}(j);
        linuxTotal = linuxTotal + linuxFilesData{i}(j);
    end
    windowsAverageFPS(i) = 1000 / (windowsTotal / numberOfDataPoints);
    linuxAverageFPS(i) = 1000 / (linuxTotal / numberOfDataPoints);
end

% Extract the max value for each yolo and blob size (cuda vs no cuda) for
% each file
windowsMaxFilesData = cell(numberOfTests, 1);
linuxMaxFilesData = cell(numberOfTests, 1);
windowsMaxFPS = zeros(numberOfTests, 1);
linuxMaxFPS = zeros(numberOfTests, 1);

% First value is the non yolo so read in automatically without checkings
windowsMaxFilesData{1} = windowsFilesData{1};
linuxMaxFilesData{1} = linuxFilesData{1};

windowsMaxFPS(1) = windowsAverageFPS(1);
linuxMaxFPS(1) = linuxAverageFPS(1);

j=2;
for i=2:2:numberOfFiles-1
    if (windowsAverageFPS(i) > windowsAverageFPS(i+1))
        windowsMaxFPS(j) = windowsAverageFPS(i);
        windowsMaxFilesData{j} = windowsFilesData{i};
    else
        windowsMaxFPS(j) = windowsAverageFPS(i+1);
        windowsMaxFilesData{j} = windowsFilesData{i+1};
    end    
    
    if (linuxAverageFPS(i) > linuxAverageFPS(i+1))
        linuxMaxFPS(j) = linuxAverageFPS(i);
        linuxMaxFilesData{j} = linuxFilesData{i};
    else
        linuxMaxFPS(j) = linuxAverageFPS(i+1);
        linuxMaxFilesData{j} = linuxFilesData{i+1};
    end
    j = j + 1;
end

% Same for jetson
jetsonFilesData = cell(numberOfTests, 1);
for i=1:numberOfTests
    jetsonFilesData{i} = importdata(strcat(jetsonFiles(i).folder, '\', jetsonFiles(i).name));
end

jetsonMaxFilesData = jetsonFilesData;
jetsonMaxFPS = zeros(numberOfTests, 1);

for i=1:numberOfTests
    jetsonTotal = 0;
    for j=2:numberOfDataPoints-1
        jetsonTotal = jetsonTotal + jetsonFilesData{i}(j);
    end
    jetsonMaxFPS(i) = 1000 / (jetsonTotal / numberOfDataPoints);
end

%% Frame Times
figure1 = figure;
set(gcf, 'Position', [100, 100, 850, 700]);

% Colour of each platform
newcolors = [0.25 0.5 1
             0.5 0.5 0.5
             0.5 0.9 0.25];
colororder(newcolors);

% Create the plot
for i=1:numberOfTests
    plot(x,windowsMaxFilesData{i}, 'LineWidth',1);
    hold on;
    plot(x,linuxMaxFilesData{i}, 'LineWidth',1);
    hold on;
    plot(x,jetsonMaxFilesData{i}, 'LineWidth',1);
    hold on;
end
xlabel('Frame Number');
ylabel('Time to compute frame (ms)');
axis tight;
ylim([0 1300]);
grid on;
title('Frame times for all platforms');
legend(' Windows 10 Desktop', ' Linux (Ubuntu 20.04) Desktop', ' Jetson Nano (4GB)', 'location','southoutside');

% Save to .png
f = gcf;
exportgraphics(f, 'All-platforms-all-frame-plots.png');

%% FPS Values
figure2 = figure;
set(gcf, 'Position',  [1050, 100, 850, 700]);

% Colour of each platform
newcolors = [0.25 0.5 1
             0.5 0.5 0.5
             0.5 0.9 0.25];
colororder(newcolors);

% Create the category and y data
x = categorical({'No YOLOv4', 'YOLOv4-tiny 288','YOLOv4-tiny 320','YOLOv4-tiny 416', 'YOLOv4-tiny 512', 'YOLOv4-tiny 608', 'YOLOv4 288','YOLOv4 320','YOLOv4 416', 'YOLOv4 512', 'YOLOv4 608'});
x = reordercats(x,{'No YOLOv4', 'YOLOv4-tiny 288','YOLOv4-tiny 320','YOLOv4-tiny 416', 'YOLOv4-tiny 512', 'YOLOv4-tiny 608', 'YOLOv4 288','YOLOv4 320','YOLOv4 416', 'YOLOv4 512', 'YOLOv4 608'});
y = [windowsMaxFPS(1), linuxMaxFPS(1), jetsonMaxFPS(1)];
for i=2:numberOfTests
    y = [y; windowsMaxFPS(i), linuxMaxFPS(i), jetsonMaxFPS(i)];
end

% Create the plot
barChart = bar(x,y);
set(barChart, {'DisplayName'}, {' Windows 10 Desktop', ' Linux (Ubuntu 20.04) Desktop', ' Jetson Nano (4GB)'}');
xlabel('YOLOv4 Type');
ylabel('Frames per seconds (FPS)');
ylim([0 100]);
grid on;
title('FPS values for all platforms');
legend();

% Display the value of each bar on top of the each bar
xtips = barChart(1).XEndPoints;
ytips = barChart(1).YEndPoints;
labels = string(int8(barChart(1).YData));
text(xtips,ytips,labels,'HorizontalAlignment','center','VerticalAlignment','bottom')

xtips = barChart(2).XEndPoints;
ytips = barChart(2).YEndPoints;
labels = string(int8(barChart(2).YData));
text(xtips,ytips,labels,'HorizontalAlignment','center','VerticalAlignment','bottom')

xtips = barChart(3).XEndPoints;
ytips = barChart(3).YEndPoints;
labels = string(int8(barChart(3).YData));
text(xtips,ytips,labels,'HorizontalAlignment','center','VerticalAlignment','bottom')

% Save to .png
f = gcf;
exportgraphics(f, 'All-platforms-all-fps-plots.png');