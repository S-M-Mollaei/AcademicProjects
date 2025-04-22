function f = jong_g(n,x0)

f = zeros(n,1);
for i = 1:n
    f(i) = 2*i*x0(i);
end
end
