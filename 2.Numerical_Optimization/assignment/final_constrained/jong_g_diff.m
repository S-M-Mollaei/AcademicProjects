function gradfx = jong_g_diff(f, x, n, k_h)

h = 10^(-k_h)*norm(x);
gradfx = zeros(n,1);
for i=1:length(x)
    xh = x;
    xh(i) = xh(i) + h;
    gradfx(i) = (f(xh) - f(x))/ h;
    % ALTERNATIVELY (no xh)
    % gradf(i) = (f([x(1:i-1); x(i)+h; x(i+1:end)]) - f(x))/h;
end
end