%% LOADING THE VARIABLES FOR THE TEST

clear 
close all
clc

load('jong_test(10^3).mat')

c1 = 1e-4;
rho = 0.8;
btmax = 50;

gamma = 1e-1;
tolx = 1e-12;

domain = 'box';

box_mins = zeros(n,1);
box_mins(:) = -5.12;

box_maxs = zeros(n,1);
box_maxs(:) = 5.12;

Pi_X = @(x) box_projection(x, box_mins, box_maxs);

FDgrad = '';
k_h = 4;
%% RUN THE CONSTR. STEEPEST DESCENT ON f

disp('**** CONSTR. STEEPEST DESCENT: n=10^3 *****')
switch FDgrad
    case ''
        disp('//////THE METHOD FOR DERIVATIVE IS EXACT//////')
    case 'fw'
        disp('//////THE METHOD FOR DERIVATIVE IS FINITE DEFFIRENCE, FORWARD//////')
    case 'c'
        disp('//////THE METHOD FOR DERIVATIVE IS FINITE DEFFIRENCE, CENTERED//////')
end
tic
[xk, fk, gradfk_norm, deltaxk_norm, k, xseq, btseq] = ...
    constr_steepest_desc_bcktrck(x0, f, gradf, ...
    kmax, tolgrad, c1, rho, btmax, gamma, tolx, Pi_X, FDgrad, k_h);
toc
disp('**** CONSTR. STEEPEST DESCENT: FINISHED *****')
disp('**** CONSTR. STEEPEST DESCENT: RESULTS *****')
disp('************************************')
disp(['xk_min: ', mat2str(min(xk)), 'xk_max: ', mat2str(max(xk)) ])
disp(['f(xk): ', num2str(fk), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k),'/',num2str(kmax), ';'])
disp(['N. of backtrackingIterationMax: ', num2str(max(btseq)), ';'])
disp(['gradient norm: ', num2str(gradfk_norm), ';'])
disp(['length of last step: ', num2str(deltaxk_norm), ';'])
disp('*****************DONE*******************')

%% PLOTS

t = linspace(0, 1, 25);


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

% Simple Plot
fig1_n = figure();
% Contour plot with curve levels for each point in xseq
[C1, ~] = contour(X1, Y1, Z1);
hold on
% plot of the points in xseq
for i = 1:n
    plot([x0(i), Pi_X_x0(i)], 'r--*')
end

for i = 1:n
    plot([Pi_X_x0(i) xseq(i, :)], 'b--*')
end
%title('h for k= ',num2str(k_h))
hold off


% Barplot of btseq
fig1_bt = figure();
bar(btseq)

%% LOADING THE VARIABLES FOR THE TEST

load('jong_test(10^4).mat')

c1 = 1e-4;
rho = 0.8;
btmax = 50;

gamma = 1e-1;
tolx = 1e-12;

domain = 'box';

box_mins = zeros(n,1);
box_mins(:) = -5.12;

box_maxs = zeros(n,1);
box_maxs(:) = 5.12;

Pi_X = @(x) box_projection(x, box_mins, box_maxs);

FDgrad = '';
k_h = 4;
%% RUN THE CONSTR. STEEPEST DESCENT ON f

disp('**** CONSTR. STEEPEST DESCENT: n=10^4 *****')
switch FDgrad
    case ''
        disp('//////THE METHOD FOR DERIVATIVE IS EXACT//////')
    case 'fw'
        disp('//////THE METHOD FOR DERIVATIVE IS FINITE DEFFIRENCE, FORWARD//////')
    case 'c'
        disp('//////THE METHOD FOR DERIVATIVE IS FINITE DEFFIRENCE, CENTERED//////')
end
tic
[xk1, fk1, gradfk_norm1, deltaxk_norm1, k1, xseq1, btseq1] = ...
    constr_steepest_desc_bcktrck(x0, f, gradf, ...
    kmax, tolgrad, c1, rho, btmax, gamma, tolx, Pi_X, FDgrad, k_h);
toc
disp('**** CONSTR. STEEPEST DESCENT: FINISHED *****')
disp('**** CONSTR. STEEPEST DESCENT: RESULTS *****')
disp('************************************')
disp(['xk_min: ', mat2str(min(xk1)), 'xk_max: ', mat2str(max(xk1)) ])
disp(['f(xk): ', num2str(fk1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k1),'/',num2str(kmax), ';'])
disp(['N. of backtrackingIterationMax: ', num2str(max(btseq1)), ';'])
disp(['gradient norm: ', num2str(gradfk_norm1), ';'])
disp(['length of last step: ', num2str(deltaxk_norm1), ';'])
disp('*****************DONE*******************')

%% LOADING THE VARIABLES FOR THE TEST

load('jong_test(10^5).mat')

c1 = 1e-4;
rho = 0.8;
btmax = 50;

gamma = 1e-1;
tolx = 1e-12;

domain = 'box';

box_mins = zeros(n,1);
box_mins(:) = -5.12;

box_maxs = zeros(n,1);
box_maxs(:) = 5.12;

Pi_X = @(x) box_projection(x, box_mins, box_maxs);

FDgrad = '';
k_h = 4;
%% RUN THE CONSTR. STEEPEST DESCENT ON f

disp('**** CONSTR. STEEPEST DESCENT: n=10^5 *****')
switch FDgrad
    case ''
        disp('//////THE METHOD FOR DERIVATIVE IS EXACT//////')
    case 'fw'
        disp('//////THE METHOD FOR DERIVATIVE IS FINITE DEFFIRENCE, FORWARD//////')
    case 'c'
        disp('//////THE METHOD FOR DERIVATIVE IS FINITE DEFFIRENCE, CENTERED//////')
end
tic
[xk2, fk2, gradfk_norm2, deltaxk_norm2, k2, xseq2, btseq2] = ...
    constr_steepest_desc_bcktrck(x0, f, gradf, ...
    kmax, tolgrad, c1, rho, btmax, gamma, tolx, Pi_X, FDgrad, k_h);
toc
disp('**** CONSTR. STEEPEST DESCENT: FINISHED *****')
disp('**** CONSTR. STEEPEST DESCENT: RESULTS *****')
disp('************************************')
disp(['xk_min: ', mat2str(min(xk2)), 'xk_max: ', mat2str(max(xk2)) ])
disp(['f(xk): ', num2str(fk2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k2),'/',num2str(kmax), ';'])
disp(['N. of backtrackingIterationMax: ', num2str(max(btseq2)), ';'])
disp(['gradient norm: ', num2str(gradfk_norm2), ';'])
disp(['length of last step: ', num2str(deltaxk_norm2), ';'])
disp('*****************DONE*******************')