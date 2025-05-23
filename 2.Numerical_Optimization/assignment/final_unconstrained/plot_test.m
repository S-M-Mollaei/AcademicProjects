%% LOADING THE VARIABLES FOR THE TEST FOR X0 = [1.2;1.2]

clear
close all
clc

c1 = 1e-4;
rho = 0.8;
btmax = 50;

load('mytest1.mat')

%% RUN THE STEEPEST DESCENT FOR X0 = [1.2;1.2]

disp('**** STEEPEST DESCENT: START WITH X0=[1.2;1.2] *****')
tic
[xk, fk, gradfk_norm, k, xseq, btseq] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('************************************')
disp(['START POINT: ', mat2str(x0)])
disp(['xk: ', mat2str(xk), ' (actual minimum: [1; 1]);'])
disp(['f(xk): ', num2str(fk), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq))])
disp(['grad_norm: ', mat2str(gradfk_norm)])
disp('************************************')

%% PLOTS

% Creation of the meshgrid for the contour-plot
[X, Y] = meshgrid(linspace(-6, 6, 500), linspace(-6, 6, 500));
% Computation of the values of f for each point of the mesh
Z = X.^2 + Y.^2;

% Plots

% Simple Plot
fig1 = figure();
% Contour plot with curve levels for each point in xseq
[C1, ~] = contour(X, Y, Z);
hold on
% plot of the points in xseq
plot([x0(1) xseq(1, :)], [x0(2) xseq(2, :)], '--*')
hold off

% More interesting Plot
fig2 = figure();
% Contour plot with curve levels for each point in xseq
% ATTENTION: actually, the mesh [X, Y] is to coarse for plotting the last
% level curves corresponding to the last point in xseq (check it zooming
% the image).
[C2, ~] = contour(X, Y, Z, f(xseq));
hold on
% plot of the points in xseq
plot([x0(1) xseq(1, :)], [x0(2) xseq(2, :)], '--*')
hold off

% Barplot of btseq
fig3 = figure();
bar(btseq)

% Much more interesting plot
fig4 = figure();
surf(X, Y, Z,'EdgeColor','none')
hold on
plot3([x0(1) xseq(1, :)], [x0(2) xseq(2, :)], [f(x0), f(xseq)], 'r--*')
hold off
