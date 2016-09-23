function [solution, optimalobjfunct] = MaxFunc(x11, x12, x21, x22, dx1, dx2)
%MaxFunc Maximizes the value of a function over a region
%   f(x, y) = 1 + x^2(y-1)^3 exp(-x-y)
%   x ranges from x11 to x12
%   y ranges from x21 to x22
%   increment x at intervals dx1
%   increment y at intervals dx2

xint = x11:dx1:x12;
yint = x21:dx2:x22;

    function y = f(x1, x2)
        y = 1 + x1^2 * (x2 - 1)^3 * exp(-x1 - x2);
    end

    function b = g(x1, x2)
        if (x2 >= log(x1) && x1 + x2 <= 6)
            b = 1;
        else
            b = 0;
        end
    end

xsize = size(xint, 2);
ysize = size(yint, 2);

func_values = nan(xsize, ysize);

for x = 1:xsize
    for y = 1:ysize
        if g(xint(x), yint(y))
            func_values(x, y) = f(xint(x), yint(y));
        end
    end
end

optimalobjfunct = max(func_values(:));

[row, col] = find(func_values(:,:) == optimalobjfunct);

solution = [xint(row), yint(col)];

end

