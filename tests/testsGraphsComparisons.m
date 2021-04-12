%% Junaid Afzal
%% Load in data
clear variables;
close all;

% Platforms
windows = 'Output for Windows 10 Desktop';
ubuntu = 'Output for Ubuntu 20.04 Desktop';
jetson = 'Output for Jetson Nano';

% File arrays
windowsFiles = dir(strcat(windows, '/'));
ubuntuFiles = dir(strcat(ubuntu, '/'));
jetsonFiles = dir(strcat(jetson, '/'));

% Consts
fileSize = length(windowsFiles);
numberOfFiles = 21;
numberOfTests = 11;
numberOfDataPoints = 1155-1; % Ignore first frame due to being very high
x = 1:1:numberOfDataPoints;

% Read in files
nonEditedWindowsFilesData = cell(numberOfFiles, 1);
nonEditedUbuntuFilesData = cell(numberOfFiles, 1);
j=1;
for i=1:fileSize
    if (windowsFiles(i).isdir == 0)
        nonEditedWindowsFilesData{j} = importdata(strcat(windowsFiles(i).folder, '/', windowsFiles(i).name));
        nonEditedUbuntuFilesData{j} = importdata(strcat(ubuntuFiles(i).folder, '/', ubuntuFiles(i).name));
        j = j + 1;
    end
end

% Remove the first frame
windowsFilesData = cell(numberOfFiles, 1);
ubuntuFilesData = cell(numberOfFiles, 1);
for i=1:numberOfFiles
    k=2;
    for j=1:numberOfDataPoints
        windowsFilesData{i}(j) = nonEditedWindowsFilesData{i}(k);
        ubuntuFilesData{i}(j) = nonEditedUbuntuFilesData{i}(k);
        k = k + 1;
    end
end

% Calculate the average FPS for each file
windowsAverageFPS = zeros(numberOfFiles, 1);
ubuntuAverageFPS = zeros(numberOfFiles, 1);
for i=1:numberOfFiles
    windowsTotal = 0;
    ubuntuTotal = 0;
    for j=1:numberOfDataPoints
        windowsTotal = windowsTotal + windowsFilesData{i}(j);
        ubuntuTotal = ubuntuTotal + ubuntuFilesData{i}(j);
    end
    windowsAverageFPS(i) = 1000 / (windowsTotal / numberOfDataPoints);
    ubuntuAverageFPS(i) = 1000 / (ubuntuTotal / numberOfDataPoints);
end

% Extract the max value for each yolo and blob size (cuda vs no cuda) for
% each file
windowsMaxFilesData = cell(numberOfTests, 1);
ubuntuMaxFilesData = cell(numberOfTests, 1);
windowsMaxFPS = zeros(numberOfTests, 1);
ubuntuMaxFPS = zeros(numberOfTests, 1);

% First value is the non yolo so read in automatically without checkings
windowsMaxFilesData{1} = windowsFilesData{1};
ubuntuMaxFilesData{1} = ubuntuFilesData{1};

windowsMaxFPS(1) = windowsAverageFPS(1);
ubuntuMaxFPS(1) = ubuntuAverageFPS(1);

j=2;
for i=2:2:numberOfFiles-1
    if (windowsAverageFPS(i) > windowsAverageFPS(i+1))
        windowsMaxFPS(j) = windowsAverageFPS(i);
        windowsMaxFilesData{j} = windowsFilesData{i};
    else
        windowsMaxFPS(j) = windowsAverageFPS(i+1);
        windowsMaxFilesData{j} = windowsFilesData{i+1};
    end    
    
    if (ubuntuAverageFPS(i) > ubuntuAverageFPS(i+1))
        ubuntuMaxFPS(j) = ubuntuAverageFPS(i);
        ubuntuMaxFilesData{j} = ubuntuFilesData{i};
    else
        ubuntuMaxFPS(j) = ubuntuAverageFPS(i+1);
        ubuntuMaxFilesData{j} = ubuntuFilesData{i+1};
    end
    j = j + 1;
end

% Same for jetson
nonEditedJetsonFilesData = cell(numberOfTests, 1);
j=1;
for i=1:numberOfTests+2
    if (jetsonFiles(i).isdir == 0)
        nonEditedJetsonFilesData{j} = importdata(strcat(jetsonFiles(i).folder, '/', jetsonFiles(i).name));
        j = j + 1;
    end
end

% Remove first frame due to very high values
jetsonMaxFilesData = cell(numberOfTests, 1);
for i=1:numberOfTests
    k=2;
    for j=1:numberOfDataPoints
        jetsonMaxFilesData{i}(j) = nonEditedJetsonFilesData{i}(k);
        k = k + 1;
    end
end

% max FPS
jetsonMaxFPS = zeros(numberOfTests, 1);
for i=1:numberOfTests
    jetsonTotal = 0;
    for j=1:numberOfDataPoints
        jetsonTotal = jetsonTotal + jetsonMaxFilesData{i}(j);
    end
    jetsonMaxFPS(i) = 1000 / (jetsonTotal / numberOfDataPoints);
end

%% Frame Times
figure1 = figure;
set(gcf, 'Position',  [100, 100, 1250, 750]);

% Colour of each platform
newcolors = [0.25 0.5 1
             0.5 0.5 0.5
             0.5 0.9 0.25];
colororder(newcolors);

% Create the plot
for i=1:numberOfTests
    plot(x,windowsMaxFilesData{i}, 'LineWidth',1);
    hold on;
    plot(x,ubuntuMaxFilesData{i}, 'LineWidth',1);
    hold on;
    plot(x,jetsonMaxFilesData{i}, 'LineWidth',1);
    hold on;
end
xlabel('Frame Number');
ylabel('Time to compute frame (ms)');
axis tight;
ylim([0 1300]);
grid on;
title('All Platforms Frame Time Plot');
legend(' Windows 10 Desktop', ' Ubuntu 20.04 Desktop', ' Jetson Nano', 'location','southoutside');

% Save to .png
f = gcf;
exportgraphics(f, 'Graphs/All Platforms Frame Time Plots.png');

%% FPS Values
figure2 = figure;
set(gcf, 'Position',  [100, 100, 1250, 750]);

% Colour of each platform
newcolors = [0.25 0.5 1
             0.5 0.5 0.5
             0.5 0.9 0.25];
colororder(newcolors);

% Create the category and y data
x = categorical({'No YOLOv4', 'YOLOv4-tiny 288','YOLOv4-tiny 320','YOLOv4-tiny 416', 'YOLOv4-tiny 512', 'YOLOv4-tiny 608', 'YOLOv4 288','YOLOv4 320','YOLOv4 416', 'YOLOv4 512', 'YOLOv4 608'});
x = reordercats(x,{'No YOLOv4', 'YOLOv4-tiny 288','YOLOv4-tiny 320','YOLOv4-tiny 416', 'YOLOv4-tiny 512', 'YOLOv4-tiny 608', 'YOLOv4 288','YOLOv4 320','YOLOv4 416', 'YOLOv4 512', 'YOLOv4 608'});
y = [windowsMaxFPS(1), ubuntuMaxFPS(1), jetsonMaxFPS(1)];

for i=7:11
    y = [y; windowsMaxFPS(i), ubuntuMaxFPS(i), jetsonMaxFPS(i)];
end

for i=2:6
    y = [y; windowsMaxFPS(i), ubuntuMaxFPS(i), jetsonMaxFPS(i)];
end

% Create the plot
barChart = bar(x,y,0.5);
set(barChart, {'DisplayName'}, {' Windows 10 Desktop', ' Ubuntu 20.04 Desktop', ' Jetson Nano'}');
xlabel('YOLOv4 Type');
ylabel('Frames per second (FPS)');
ylim([0 100]);
grid on;
title('All Platforms FPS Plot');
legend();

% Display the value of each bar on top of the each bar
xtips = barChart(1).XEndPoints;
ytips = barChart(1).YEndPoints;
labels = strings([1,11]);
for i=1:11
    labels(i) = num2str(barChart(1).YData(i), '%.1f');
end    
text(xtips,ytips,labels,'HorizontalAlignment','center','VerticalAlignment','bottom')

xtips = barChart(2).XEndPoints;
ytips = barChart(2).YEndPoints;
labels = strings([1,11]);
for i=1:11
    labels(i) = num2str(barChart(2).YData(i), '%.1f');
end
text(xtips,ytips,labels,'HorizontalAlignment','center','VerticalAlignment','bottom')

xtips = barChart(3).XEndPoints;
ytips = barChart(3).YEndPoints;
labels = strings([1,11]);
for i=1:11
    labels(i) = num2str(barChart(3).YData(i), '%.1f');
end
text(xtips,ytips,labels,'HorizontalAlignment','center','VerticalAlignment','bottom')

% Save to .png
f = gcf;
exportgraphics(f, 'Graphs/All Platforms FPS Plots.png');

clear variables;
close all;