clear; clc;
rng('default'); 
max_iter = 100;
eps = 1e-3;

%% generate synthetic data
% ratio = 5;
% m = 100*ratio;
% n = 100*ratio;
% k = 30*ratio;
% rate = .2;

% W0 = rand(m,k); W0(rand(m,k)<rate)=0;
% H0 = rand(k,n); H0(rand(k,k)<rate)=0;
% V = W0 * H0;

%% Load text datas
% load('/Users/gaotx/Documents/MATLAB/nmf/real_data/News20.mat'); max_iter=100;
% load('/Users/gaotx/Documents/MATLAB/nmf/real_data/MNIST_28x28.mat'); max_iter=23;
% load('/Users/gaotx/Documents/MATLAB/nmf/real_data/TDT2.mat'); % maxIter=1000;
% load('/Users/gaotx/Documents/MATLAB/nmf/real_data/Reuters21578.mat'); % maxIter=1000;

%% Load face datas
% load('/Users/gaotx/Documents/MATLAB/nmf/real_data/UMist_112x92.mat'); max_iter=1000;
% load('/Users/gaotx/Documents/MATLAB/nmf/real_data/YaleB_32x32.mat'); % max_iter=1000;
% load('/Users/gaotx/Documents/MATLAB/nmf/real_data/ORL_32x32.mat'); % max_iter=1000;
% load('/Users/gaotx/Documents/MATLAB/nmf/real_data/COIL100.mat'); % max_iter=1000;

%% Load new sensor data
load('/Users/gaotx/Documents/MATLAB/nmf/real_data/NIR_face_28x21.mat');

%% For read data, make sure command all synthetic data part
% k = length(unique(gnd));
k = 197; % for NIR_face
V = fea'; clear fea;
V = full(double(V));
V = V/max(max(V));
[m,n] = size(V);
%% Subsets of news20
% k = 2;
% index = gnd == 2 | gnd == 9;
% V = full(fea(index,:)); clear fea; V = V';
% V = V/(max(max(V)));
% [m,n] = size(V);

times_Mult =[];
times_admm =[];
times_pivot = [];
times_CD =[];
warning('off');
for trial = 1:1
%% init
rng('default'); 
W = abs(rand(m,k));
H = abs(rand(k,n));

%% Multiplicative update rule
tic
[W1,H1,obs_Mult, time_Mult] = KL_Mult(V,W,H,max_iter,eps);
times_Mult = [times_Mult toc];
fprintf("obs_Mult(1) = %f, obs_Mult(end) = %f, iter=%d, time=%f\n", obs_Mult(1),obs_Mult(end), length(obs_Mult), time_Mult(end));

%% ADMM
beta = 1; % set beta=1 for KL divergence, beta=0 for IS divergence
rho = 1; % ADMM parameter
KL = 1;

tic
[W3,H3,obs_admm, time_admm] = nmf_admm(V, W, H, beta, rho, [],max_iter, KL, eps);
times_admm = [times_admm toc];
fprintf("obs_admm(1) = %f, obs_admm(end) = %f, iter=%d, time=%f\n", obs_admm(1),obs_admm(end), length(obs_admm), time_admm(end));
%% Pivot
% rho = 1;
tic
[W4,H4,obs_pivot, time_pivot] = nmf_pivot(V, W, H, rho, max_iter, eps);
times_pivot = [times_pivot toc];
fprintf("obs_pivot(1) = %f, obs_pivot(end) = %f, iter=%d, time=%f\n", obs_pivot(1),obs_pivot(end), length(obs_pivot), time_pivot(end));

%% KL_Bregman
% tic
% [W2,H2,obs_CD] = KL_Bregman(V,W,H,max_iter, eps);
% [w h obs_CD time0] = KLnmf(V,k,max_iter(1),W',H, 1);
% times_CD = [times_CD toc];
% fprintf("obs_CD(1) = %f, obs_CD(end) = %f\n", obs_CD(1),obs_CD(end));
end

% fprintf(['sum(times_Mult)=%.4f, sum(times_admm)=%.4f,'...
%     'sum(times_CD)=%.4f, sum(times_pivot)=%.4f\n'],sum(times_Mult)/trial,...
%     sum(times_admm)/trial,sum(times_CD)/trial, sum(times_pivot)/trial);

fprintf(['sum(times_Mult)=%.4f, sum(times_admm)=%.4f,sum(times_pivot)=%.4f\n'],sum(times_Mult)/trial,...
    sum(times_admm)/trial,sum(times_pivot)/trial);
%% Plots
lw = 1.5;
fs = 14;
ms = 6;

logobs_Mult = log10(obs_Mult);
% logobs_CD = log10(obs_CD);
logobs_admm = log10(obs_admm);
logobs_pivot = log10(obs_pivot);

f1 = figure; hold on;
plot(logobs_Mult,'linestyle','-','linewidth',lw,'color', 'b','marker','>','MarkerFaceColor','b','markersize',ms,'MarkerIndices',1:round(length(logobs_Mult)/10):length(logobs_Mult));
plot(logobs_admm,'linestyle','--','linewidth',lw,'color','c','marker','d','MarkerFaceColor','c','markersize',ms,'MarkerIndices',1:round(length(logobs_admm)/10):length(logobs_admm));
% plot(logobs_CD,'linestyle','-.','linewidth',lw,'color','r','marker','o','MarkerFaceColor','r','markersize',ms,'MarkerIndices',1:round(length(logobs_CD)/10):length(logobs_CD));
plot(logobs_pivot,'linestyle','--','linewidth',lw,'color','r','marker','o','MarkerFaceColor','r','markersize',ms,'MarkerIndices',1:round(length(logobs_pivot)/10):length(logobs_pivot));

legend('Multiplicative','ADMM', 'Block');
% legend('Multiplicative','ADMM','Coordinate Descent', 'Pivot');
xlabel('iteration');
ylabel('Objective values in log scale');
title('KL Distance.');
hold off;

output_folder = 'xinyao';
filename1 = fullfile(output_folder, 'NIR_iter.eps');
set(f1, 'PaperUnits', 'inches', 'PaperPosition', [0 0 6 4]);
print(filename1, '-depsc', '-r300');

%% times
f2 = figure; hold on;
plot(time_Mult, logobs_Mult,'linestyle','-','linewidth',lw,'color', 'b','marker','>','MarkerFaceColor','b','markersize',ms,'MarkerIndices',1:round(length(logobs_Mult)/10):length(logobs_Mult));
plot(time_admm, logobs_admm,'linestyle','--','linewidth',lw,'color','c','marker','d','MarkerFaceColor','c','markersize',ms,'MarkerIndices',1:round(length(logobs_admm)/10):length(logobs_admm));
% plot(logobs_CD,'linestyle','-.','linewidth',lw,'color','r','marker','o','MarkerFaceColor','r','markersize',ms,'MarkerIndices',1:round(length(logobs_CD)/10):length(logobs_CD));
plot(time_pivot, logobs_pivot,'linestyle','--','linewidth',lw,'color','r','marker','o','MarkerFaceColor','r','markersize',ms,'MarkerIndices',1:round(length(logobs_pivot)/10):length(logobs_pivot));

legend('Multiplicative','ADMM', 'Block');
% legend('Multiplicative','ADMM','Coordinate Descent', 'Pivot');
xlabel('time(seconds)');
ylabel('Objective values in log scale');
title('KL Distance.');
hold off;

filename2 = fullfile(output_folder, 'NIR_time.eps');
set(f2, 'PaperUnits', 'inches', 'PaperPosition', [0 0 6 4]);
print(filename2, '-depsc', '-r300');

%% Save plots
% output_folder = 'xinyao';
% output_filename = fullfile(output_folder, 'NIR_iter.eps');
% output_filename = fullfile(output_folder, 'NIR_time.eps');
% print(output_filename, '-depsc', '-r300');

% savefig(h,'NIR_iter.fig');
% print('NIR_iter','-depsc');
% print('NIR_time','-depsc');
% print(gcf, '-dpdf', 'pivot.pdf');

% Save both figures in a single file
% Save both figures in the "my_figures" folder
foldername = 'xinyao';
if ~exist(foldername, 'dir')
    mkdir(foldername);
end
savefig(f1, fullfile(foldername, 'fig1.fig'));
savefig(f2, fullfile(foldername, 'fig2.fig'));