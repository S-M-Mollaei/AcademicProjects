%% LOADING THE VARIABLES FOR THE TEST

clear 
close all
clc

load('jong_test.mat')

c1 = 1e-4;
rho = 0.8;
btmax = 50;

FDgrad = 'c';
FDHess = 'c';

h = sqrt(eps);

%% RUN THE NEWTON ON f1

disp('**** NEWTON: START f1 *****')

[xk_n1, fk_n1, gradfk_norm_n1, k_n1, xseq_n1, btseq_n1] = ...
    newton_general(x0, f1, gradf1, Hessf1, kmax, ...
    tolgrad, c1, rho, btmax, FDgrad, FDHess, h);
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_n1), ' (actual minimum: [0; 0]);'])
disp(['f(xk): ', num2str(fk_n1), ' (actual min. value: 5);'])
disp(['N. of Iterations: ', num2str(k_n1),'/',num2str(kmax), ';'])
disp('************************************')

disp('---')

%% RUN THE NEWTON ON f2

disp('**** NEWTON: START f2 *****')

[xk_n2, fk_n2, gradfk_norm_n2, k_n2, xseq_n2, btseq_n2] = ...
    newton_general(x0, f2, gradf2, Hessf2, kmax, ...
    tolgrad, c1, rho, btmax, FDgrad, FDHess, h);
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_n2), ' (actual minimum: [1; 1]);'])
disp(['f(xk): ', num2str(fk_n2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_n2),'/',num2str(kmax), ';'])
disp('************************************')

disp('---')

%% RUN THE NEWTON ON f3

disp('**** NEWTON: START f3 *****')

[xk_n3, fk_n3, gradfk_norm_n3, k_n3, xseq_n3, btseq_n3] = ...
    newton_general(x0, f3, gradf3, Hessf3, kmax, ...
    tolgrad, c1, rho, btmax, FDgrad, FDHess, h);
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_n3), ' (actual minima: [3; 2], ~[-2.805,3.131]), ~[-3.779,-3.283], ~[3.584, -1.848];'])
disp(['f(xk): ', num2str(fk_n3), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_n3),'/',num2str(kmax), ';'])
disp('************************************')



%% PLOTS

% Creation of the meshgrid for the contour-plot
[X1, Y1] = meshgrid(linspace(-6, 6, 500), linspace(-6, 6, 500));

% Creation of the meshgrid for the contour-plot
[X2, Y2] = meshgrid(linspace(-6, 6, 500), linspace(-6, 25, 500));
% Computation of the values of f for each point of the mesh

% Computation of the values of f for each point of the mesh
Z1 = X1.^2 + 4*Y1.^2 + 5;
Z2 = 100*(Y2-X2.^2).^2+(1-X2).^2;
Z3 = (X1.^2+Y1-11).^2+(X1+Y1.^2-7).^2;

% Plots

% Simple Plot
fig1_n = figure();
% Contour plot with curve levels for each point in xseq
[C1, ~] = contour(X1, Y1, Z1);
hold on
% plot of the points in xseq
plot([x0(1) xseq_n1(1, :)], [x0(2) xseq_n1(2, :)], '--*')
hold off
title('Newton - x^2 + 4y^2 + 5')

% Barplot of btseq
fig1_bt = figure();
bar(btseq_n1)
title('Newton - x^2 + 4y^2 + 5')

% Much more interesting plot
fig1_surf = figure();
surf(X1, Y1, Z1,'EdgeColor','none')
hold on
plot3([x0(1) xseq_n1(1, :)], [x0(2) xseq_n1(2, :)], [f1(x0), f1(xseq_n1)], 'r--*')
hold off
title('Newton - x^2 + 4y^2 + 5')

% Simple Plot
fig2_n = figure();
% Contour plot with curve levels for each point in xseq
[C2, ~] = contour(X2, Y2, Z2);
hold on
% plot of the points in xseq
plot([x0(1) xseq_n2(1, :)], [x0(2) xseq_n2(2, :)], '--*')
hold off
title('Newton - Rosenbrock')

% Barplot of btseq
fig2_bt = figure();
bar(btseq_n2)
title('Newton - Rosenbrock')

% Much more interesting plot
fig2_surf = figure();
surf(X2, Y2, Z2,'EdgeColor','none')
hold on
plot3([x0(1) xseq_n2(1, :)], [x0(2) xseq_n2(2, :)], [f2(x0), f2(xseq_n2)], 'r--*')
hold off
title('Newton - Rosenbrock')

% Simple Plot
fig3_n = figure();
% Contour plot with curve levels for each point in xseq
[C3, ~] = contour(X1, Y1, Z3);
hold on
% plot of the points in xseq
plot([x0(1) xseq_n3(1, :)], [x0(2) xseq_n3(2, :)], '--*')
hold off
title('Newton - Himmelblau')

% Barplot of btseq
fig3_bt = figure();
bar(btseq_n3)
title('Newton - Himmelblau')

% Much more interesting plot
fig3_surf = figure();
surf(X1, Y1, Z3,'EdgeColor','none')
hold on
plot3([x0(1) xseq_n3(1, :)], [x0(2) xseq_n3(2, :)], [f3(x0), f3(xseq_n3)], 'r--*')
hold off
title('Newton - Himmelblau')

