function [W, H, obs, times] = nmf_pivot(V, W, H, rho, max_iter, eps)

[m,k] = size(W);
X = W*H;
Wplus = W;
% Hplus = H;
alphaX = zeros(size(X));
alphaW = zeros(size(W));
% alphaH = zeros(size(H));

obs = comp_obj(V,W,H);
times = toc;
% tic;
for iter = 1:max_iter
    H = nnlsm_blockpivot(W, X + alphaX/rho);
%     W = nnlsm_blockpivot(H', (X+alphaX/rho)')';
    W = (X*H' + Wplus + (alphaX*H' - alphaW)/rho) / (H*H' + eye(k));
    
    X_ap = W*H;
    b = rho*X_ap - alphaX - 1;
    X = (b + sqrt(b.^2 + 4*rho*V))/(2*rho);
    
    % update for H_+ and W_+
%     Hplus = max(H + 1/rho*alphaH, 0);
    Wplus = max(W + 1/rho*alphaW, 0);

    % update for dual variables
    alphaX = alphaX + rho*(X - X_ap);
%     alphaH = alphaH + rho*(H - Hplus);
    alphaW = alphaW + rho*(W - Wplus);
    
    obs = [obs comp_obj(V,W,H)];
    
    times = [times toc-times(1)];
    
    if obs(end) <= eps
        break;
    end 
end
% toc;

end

% figure;
% plot(obs);
% [W3,H3,obs_admm] = nmf_admm(V, W0, H0, beta, rho, [],max_iter, KL, eps);
% figure;
% plot(obs_admm);
function obj = comp_obj(A,W,H)
WH = W*H;
% obj = sum(sum(A.*log(A+1e-10)-A.*log(WH+1e-10)-A+WH));
obj = norm(A-WH)/norm(A);
end