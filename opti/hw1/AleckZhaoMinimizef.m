function [solution, optimalobjfunct] = AleckZhaoMinimizef(a, b, h)
%AleckZhaoMinimizef Minimizes function over interval
%   f(x) = (x-1)^2 sin x is the function
%   interval is [a, b]
%   brute force at increments of h

interval = a:h:b;

    function y = f(x) 
        y = (x-1)^2 * sin(x);
    end

func_values = arrayfun(@f, interval);

[optimalobjfunct, index] = min(func_values);

solution = interval(index);

end

