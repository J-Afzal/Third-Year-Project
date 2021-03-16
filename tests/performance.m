%% Junaid Afzal
%% Load in data
clear variables;
close all;

% Platforms
% platform = 'Windows 10 Desktop';
% platform = 'Linux (Ubuntu 20.04) Desktop';
% platform = 'Jetson Nano (4GB)';

% File arrays
files = dir(strcat(platform, '/*.txt'));

% Consts
numberOfFiles = length(files);
numberOfDataPoints = 1155;
x = 1:1:numberOfDataPoints;

% Read in files
filesData = cell(numberOfFiles, 1);
for i=1:numberOfFiles
    filesData{i} = importdata(strcat(files(i).folder, '\', files(i).name));
end

% Calculate the average FPS
averageFPS = zeros(numberOfFiles, 1);
for i=1:numberOfFiles
    total = 0;
    for j=2:numberOfDataPoints-1
        total = total + filesData{i}(j);
    end
    averageFPS(i) = 1000 / (total / numberOfDataPoints);
end

%% All frame plots
figure1 = figure;
set(gcf, 'Position',  [100, 100, 850, 700]);

% Set the colormap to green for CUDA and purple for
if (strcmp(platform, 'Jetson Nano (4GB)'))
    newcolors = [0 0.7 0];
    colororder(newcolors);
else
    newcolors = [0 0 0
                 0.7 0 0
                 0 0.7 0
                 0.7 0 0
                 0 0.7 0
                 0.7 0 0
                 0 0.7 0
                 0.7 0 0
                 0 0.7 0
                 0.7 0 0
                 0 0.7 0
                 0.7 0 0
                 0 0.7 0
                 0.7 0 0
                 0 0.7 0
                 0.7 0 0
                 0 0.7 0
                 0.7 0 0
                 0 0.7 0
                 0.7 0 0
                 0 0.7 0];
    colororder(newcolors);
end

for i=1:numberOfFiles
    plot(x,filesData{i}, 'LineWidth',0.9);
    hold on;
end

xlabel('Frame Number');
ylabel('Time to compute frame (ms)');
axis tight;
ylim([0 300]);
grid on;
title(strcat('Frame times for ', {' '}, platform));
if (strcmp(platform, 'Jetson Nano (4GB)'))
    legend('No YOLOv4 and YOLOv4 with CUDA', 'location','southoutside');
else
    legend('No YOLOv4', ' YOLOv4 without CUDA', 'YOLOv4 with CUDA', 'location','southoutside');
end

% Save to .png
f = gcf;
exportgraphics(f, strcat(platform, '/All-frame-plots.png'));

%% All fps plots
figure2 = figure;
set(gcf, 'Position',  [1050, 100, 850, 700]);

% Set the colormap to green for CUDA and purple for
if (strcmp(platform, 'Jetson Nano (4GB)'))
    newcolors = [0 0.7 0];
    colororder(newcolors);
else
    newcolors = [0.7 0 0
             0 0.7 0];
    colororder(newcolors);
end

% Create the category and y data
x = categorical({'No YOLOv4', 'YOLOv4-tiny 288','YOLOv4-tiny 320','YOLOv4-tiny 416', 'YOLOv4-tiny 512', 'YOLOv4-tiny 608', 'YOLOv4 288','YOLOv4 320','YOLOv4 416', 'YOLOv4 512', 'YOLOv4 608'});
x = reordercats(x,{'No YOLOv4', 'YOLOv4-tiny 288','YOLOv4-tiny 320','YOLOv4-tiny 416', 'YOLOv4-tiny 512', 'YOLOv4-tiny 608', 'YOLOv4 288','YOLOv4 320','YOLOv4 416', 'YOLOv4 512', 'YOLOv4 608'});

if (strcmp(platform, 'Jetson Nano (4GB)'))
y = [averageFPS(1)];
    for i=2:11
        y = [y; averageFPS(i)];
    end
else
    y = [NaN, averageFPS(1)];
    for i=2:2:numberOfFiles
        y = [y; averageFPS(i),averageFPS(i+1)];
    end
end

% Create the plot
barChart = bar(x,y);
set(barChart, {'DisplayName'}, {' with CUDA'}');
xlabel('YOLOv4 Type');
ylabel('Frames per seconds (FPS)');
if (strcmp(platform, 'Jetson Nano (4GB)'))
    ylim([0 15]);
else
    ylim([0 100]);
end
grid on;
title(strcat('FPS values for', {' '}, platform));
legend();

% Display the value of each bar on top of the each bar
if (strcmp(platform, 'Jetson Nano (4GB)'))
    xtips = barChart(1).XEndPoints;
    ytips = barChart(1).YEndPoints;
    labels = string(barChart(1).YData);
    text(xtips,ytips,labels,'HorizontalAlignment','center','VerticalAlignment','bottom')
else
    xtips = barChart(1).XEndPoints;
    ytips = barChart(1).YEndPoints;
    labels = string(int8(barChart(1).YData));
    text(xtips,ytips,labels,'HorizontalAlignment','center','VerticalAlignment','bottom')

    xtips = barChart(2).XEndPoints;
    ytips = barChart(2).YEndPoints;
    labels = string(int8(barChart(2).YData));
    text(xtips,ytips,labels,'HorizontalAlignment','center','VerticalAlignment','bottom')
end

% Save to .png
f = gcf;
exportgraphics(f, strcat(platform, '/All-fps-plots.png'));
