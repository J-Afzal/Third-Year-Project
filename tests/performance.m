%% Junaid Afzal
%% Load in data
clear variables;
close all;

platform = 'windows';
numberOfDataPoints = 1155;

files = dir(strcat(platform, '/*.txt'));
numberOfFiles = length(files);
filesData = cell(numberOfFiles, 1);
x = 1:1:numberOfDataPoints;

% Read in files
for i=1:numberOfFiles
    filesData{i} = importdata(strcat(files(i).folder, '\', files(i).name));
end

% Calculate the averages
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

for i=1:numberOfFiles
    plot(x,filesData{i});
    hold on;
end

xlabel('Frame Number');
ylabel('Time to compute frame (ms)');
axis tight;
ylim([0 300]);
grid on;
title(strcat('All frame times for ', {' '}, platform));
legend('No YOLOv4', ' YOLOv4 without CUDA', 'YOLOv4 with CUDA', 'location','southoutside');

% Save to .png
f = gcf;
exportgraphics(f, strcat(platform, '/All-frame-plots.png'));

%% All fps plots
figure2 = figure;
set(gcf, 'Position',  [1050, 100, 850, 700]);

% Set the colormap to green for CUDA and purple for 
newcolors = [0.7 0 0
             0 0.7 0];         
colororder(newcolors);

% Create the category and y data
x = categorical({'No YOLOv4', 'YOLOv4-tiny 288','YOLOv4-tiny 320','YOLOv4-tiny 416', 'YOLOv4-tiny 512', 'YOLOv4-tiny 608', 'YOLOv4 288','YOLOv4 320','YOLOv4 416', 'YOLOv4 512', 'YOLOv4 608'});
x = reordercats(x,{'No YOLOv4', 'YOLOv4-tiny 288','YOLOv4-tiny 320','YOLOv4-tiny 416', 'YOLOv4-tiny 512', 'YOLOv4-tiny 608', 'YOLOv4 288','YOLOv4 320','YOLOv4 416', 'YOLOv4 512', 'YOLOv4 608'});
y = [NaN, averageFPS(1)];
for i=2:2:numberOfFiles
    y = [y; averageFPS(i),averageFPS(i+1)];
end

% Create the plot
barChart = bar(x,y);
set(barChart, {'DisplayName'}, {' without CUDA',' with CUDA'}');
xlabel('YOLOv4 Type');
ylabel('Frames per seconds (FPS)');
ylim([0 100]);
grid on;
title(strcat('All frame times for', {' '}, platform));
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

% Save to .png
f = gcf;
exportgraphics(f, strcat(platform, '/All-fps-plots.png'));