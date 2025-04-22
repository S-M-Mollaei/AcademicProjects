clc
clear all

s = load('mytest1.mat')

alpha=s.alpha;
kmax=s.kmax;
tolgrad=s.tolgrad;

f=s.f;
gradf=s.gradf;
Hessf=s.Hessf;
x0=s.x0;



n = 10^3;
x0 = zeros(n,1);
x0(:) = -1;

f = @(x) (Gregory_f ( n, x ));
gradf = @(x) (Gregory_g ( n, x ));
Hessf = @(x) (Gregory_h ( n, x ));

save('Gregory_test(-1).mat','Hessf','alpha','f','gradf','kmax','tolgrad','x0', 'n')