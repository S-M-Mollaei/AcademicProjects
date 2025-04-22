function f = jong_h(n, x)

f=spdiags([],[],n,n);
for i = 1:n
    f(i,i) =  2*i;
end
end