%% LOADING THE VARIABLES FOR THE TEST

clear 
close all
clc

load('test_functions2.mat')

c1 = 1e-4;
rho = 0.8;
btmax = 50;

gamma = 1e-1;
tolx = 1e-12;

% domain = 'sphere';
domain = 'box';

sphere_c = [-3; -3];
sphere_r = 1.5;

box_mins = [-4.5; -4.5];
box_maxs = [-1.5; -1.5];

switch domain
    case 'sphere'
        Pi_X = @(x) sphere_projection(x, sphere_c, sphere_r);
    case 'box'
        Pi_X = @(x) box_projection(x, box_mins, box_maxs);
    otherwise
        Pi_X = @(x) sphere_projection(x, sphere_c, sphere_r);
end



%% RUN THE NEWTON ON f1

disp('**** CONSTR. STEEPEST DESCENT: START f1 *****')

[xk_n1, fk_n1, gradfk_norm_n1, deltaxk_norm_n1, k_n1, xseq_n1, btseq_n1] = ...
    constr_steepest_desc_bcktrck(x0, f1, gradf1, ...
    kmax, tolgrad, c1, rho, btmax, gamma, tolx, Pi_X);
disp('**** CONSTR. STEEPEST DESCENT: FINISHED *****')
disp('**** CONSTR. STEEPEST DESCENT: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_n1), ' (actual minimum: [0; 0]);'])
disp(['f(xk): ', num2str(fk_n1), ' (actual min. value: 5);'])
disp(['N. of Iterations: ', num2str(k_n1),'/',num2str(kmax), ';'])
disp(['gradient norm: ', num2str(gradfk_norm_n1), ';'])
disp(['length of last step: ', num2str(deltaxk_norm_n1), ';'])
disp('************************************')

disp('---')

%% RUN THE NEWTON ON f2

disp('**** CONSTR. STEEPEST DESCENT: START f2 *****')

[xk_n2, fk_n2, gradfk_norm_n2, deltaxk_norm_n2, k_n2, xseq_n2, btseq_n2] = ...
    constr_steepest_desc_bcktrck(x0, f2, gradf2, ...
    kmax, tolgrad, c1, rho, btmax, gamma, tolx, Pi_X);
disp('**** CONSTR. STEEPEST DESCENT: FINISHED *****')
disp('**** CONSTR. STEEPEST DESCENT: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_n2), ' (actual minimum: [1; 1]);'])
disp(['f(xk): ', num2str(fk_n2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_n2),'/',num2str(kmax), ';'])
disp(['gradient norm: ', num2str(gradfk_norm_n2), ';'])
disp(['length of last step: ', num2str(deltaxk_norm_n2), ';'])
disp('************************************')

disp('---')

%% RUN THE NEWTON ON f3

disp('**** CONSTR. STEEPEST DESCENT: START f2 *****')

[xk_n3, fk_n3, gradfk_norm_n3, deltaxk_norm_n3, k_n3, xseq_n3, btseq_n3] = ...
    constr_steepest_desc_bcktrck(x0, f3, gradf3, ...
    kmax, tolgrad, c1, rho, btmax, gamma, tolx, Pi_X);
disp('**** CONSTR. STEEPEST DESCENT: FINISHED *****')
disp('**** CONSTR. STEEPEST DESCENT: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_n3), ' (actual minima: [3; 2], ~[-2.805,3.131]), ~[-3.779,-3.283], ~[3.584, -1.848]);'])
disp(['f(xk): ', num2str(fk_n3), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_n3),'/',num2str(kmax), ';'])
disp(['gradient norm: ', num2str(gradfk_norm_n3), ';'])
disp(['length of last step: ', num2str(deltaxk_norm_n3), ';'])
disp('************************************')

disp('---')



%% PLOTS

% Creation of the data to plot the domain boundaries
switch domain
    case 'sphere'
        t = linspace(0, 2 * pi, 100);
        dom_xy = sphere_c + sphere_r * [cos(t); sin(t)];
        f1_z = f1(dom_xy); 
        f2_z = f2(dom_xy); 
        f3_z = f3(dom_xy); 
    case 'box'
        t = linspace(0, 1, 25);
        dom_xy_1 = box_mins + t .* ([box_mins(1); box_maxs(2)] - box_mins);
        dom_xy_2 = [box_mins(1); box_maxs(2)] + t .* (box_maxs - [box_mins(1); box_maxs(2)]);
        dom_xy_3 = box_maxs + t .* ([box_maxs(2); box_mins(1)] - box_maxs);
        dom_xy_4 = [box_maxs(2); box_mins(1)] + t .* (box_mins - [box_maxs(2); box_mins(1)]);
        
        dom_xy = [dom_xy_1, dom_xy_2, dom_xy_3, dom_xy_4];
        f1_z = f1(dom_xy); 
        f2_z = f2(dom_xy); 
        f3_z = f3(dom_xy); 
    otherwise
        t = linspace(0, 2 * pi, 100);
        dom_xy = sphere_c + sphere_r * [cos(t); sin(t)];
        f1_z = f1(dom_xy); 
        f2_z = f2(dom_xy); 
        f3_z = f3(dom_xy); 
end

% Projection of the starting point
Pi_X_x0 = Pi_X(x0);

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
plot([x0(1), Pi_X_x0(1)], [x0(2), Pi_X_x0(2)], 'r--*')
plot([Pi_X_x0(1) xseq_n1(1, :)], [Pi_X_x0(2) xseq_n1(2, :)], 'b--*')
plot(dom_xy(1, :), dom_xy(2, :), 'k')
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
plot3([x0(1) Pi_X_x0(1)], [x0(2) Pi_X_x0(2)], [f1(x0), f1(Pi_X_x0)], 'y--*')
plot3([Pi_X_x0(1) xseq_n1(1, :)], [Pi_X_x0(2) xseq_n1(2, :)], [f1(Pi_X_x0), f1(xseq_n1)], 'r--*')
plot3(dom_xy(1, :), dom_xy(2, :), f1_z, 'k')
hold off
title('Newton - x^2 + 4y^2 + 5')

% Simple Plot
fig2_n = figure();
% Contour plot with curve levels for each point in xseq
[C2, ~] = contour(X2, Y2, Z2);
hold on
% plot of the points in xseq
plot([x0(1), Pi_X_x0(1)], [x0(2), Pi_X_x0(2)], 'r--*')
plot([Pi_X_x0(1) xseq_n2(1, :)], [Pi_X_x0(2) xseq_n2(2, :)], 'b--*')
plot(dom_xy(1, :), dom_xy(2, :), 'k')
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
plot3([x0(1) Pi_X_x0(1)], [x0(2) Pi_X_x0(2)], [f2(x0), f2(Pi_X_x0)], 'y--*')
plot3([Pi_X_x0(1) xseq_n2(1, :)], [Pi_X_x0(2) xseq_n2(2, :)], [f2(Pi_X_x0), f2(xseq_n2)], 'r--*')
plot3(dom_xy(1, :), dom_xy(2, :), f2_z, 'k')
hold off
title('Newton - Rosenbrock')

% Simple Plot
fig3_n = figure();
% Contour plot with curve levels for each point in xseq
[C3, ~] = contour(X1, Y1, Z3);
hold on
% plot of the points in xseq
plot([x0(1), Pi_X_x0(1)], [x0(2), Pi_X_x0(2)], 'r--*')
plot([Pi_X_x0(1) xseq_n3(1, :)], [Pi_X_x0(2) xseq_n3(2, :)], 'b--*')
plot(dom_xy(1, :), dom_xy(2, :), 'k')
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
plot3([x0(1) Pi_X_x0(1)], [x0(2) Pi_X_x0(2)], [f3(x0), f3(Pi_X_x0)], 'y--*')
plot3([Pi_X_x0(1) xseq_n3(1, :)], [Pi_X_x0(2) xseq_n3(2, :)], [f3(Pi_X_x0), f3(xseq_n3)], 'r--*')
plot3(dom_xy(1, :), dom_xy(2, :), f3_z, 'k')
hold off
title('Newton - Himmelblau')

