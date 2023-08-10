function [W,H,obs, times] = KL_Mult(A,W,H,max_iter,eps)
rng('default');

obs = comp_obj(A,W,H);
times = toc;
for iter = 1:max_iter
    % Update H
    H = update_H(A,W,H);
    
    % Update W
    W = update_H(A',H',W')';
    
    obs = [obs comp_obj(A,W,H)];
    times = [times toc-times(1)];
%     obs = [obs comp_obj(A,W,H)/obs(1)];
    if obs(end) <= eps
        break;
    end
end


end

function H = update_H(A,W,H)
    Wt1 = sum(W,1);
    WH = W*H + 1e-5;
    
    H = H./Wt1' .* (W'*(A./WH));
end

function obj = comp_obj(A,W,H)
WH = W*H;
% obj = sum(sum(A.*log(A+1e-10)-A.*log(WH+1e-10)-A+WH));
obj = norm(A-WH)/norm(A);
end

