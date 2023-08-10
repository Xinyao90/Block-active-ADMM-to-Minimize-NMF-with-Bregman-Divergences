function [W, H, obs, times] = nmf_admm(V, W, H, beta, rho, fixed, max_iter, KL,eps)
% nmf_admm(V, W, H, rho, inds)
% 
% Implements NMF algorithm described in:
%   D.L. Sun and C. Févotte, "Alternating direction method of multipliers 
%      for non-negative matrix factorization with the beta divergence", ICASSP 2014.
%
% inputs
%    V: matrix to factor 
%    W, H: initializations for W and H
%    beta: parameter of beta divergence 
%          (only beta=0 (IS) and beta=1 (KL) are supported)
%    rho: ADMM smothing parameter
%    fixed: a vector containing the indices of the basis vectors in W to
%           hold fixed (e.g., when W is known a priori)
%
% outputs
%    W, H: factorization such that V \approx W*H

    % determine dimensions
    [m,n] = size(V);
    [~,k] = size(W);

    % set defaults
    if nargin<5, rho=1; end
    if nargin<6, fixed=[]; end
    
    % get the vector of indices to update
    free = setdiff(1:k, fixed);
    
    % initializations for other variables
    X = W*H;
    Wplus = W;
    Hplus = H;
    alphaX = zeros(size(X));
    alphaW = zeros(size(W));
    alphaH = zeros(size(H));
    
    if KL
        obs = comp_obj(V,W,H);
    else
        obs = comp_obj_IS(V,W,H);
    end
    
    times = toc;
    for iter=1:max_iter
        
        % update for H
        H = (W'*W + eye(k)) \ (W'*X + Hplus + 1/rho*(W'*alphaX - alphaH));
        
        % update for W
        P = H*H' + eye(k);
        Q = H*X' + Wplus' + 1/rho*(H*alphaX' - alphaW');
        W(:,free) = ( P(:,free) \ (Q - P(:,fixed)*W(:,fixed)') )';
        
        % update for X (this is the only step that depends on beta)
        X_ap = W*H;
        if beta==1
            b = rho*X_ap - alphaX - 1;
            X = (b + sqrt(b.^2 + 4*rho*V))/(2*rho);
        elseif beta==0
            A = alphaX/rho - X_ap;
            B = 1/(3*rho) - A.^2/9;
            C = - A.^3/27 + A/(6*rho) + V/(2*rho);
            D = B.^3 + C.^2;

            X(D>=0) = nthroot(C(D>=0)+sqrt(D(D>=0)),3) + ...
                nthroot(C(D>=0)-sqrt(D(D>=0)),3) - ...
                A(D>=0)/3;

            phi = acos(C(D<0) ./ ((-B(D<0)).^1.5));
            X(D<0) = 2*sqrt(-B(D<0)).*cos(phi/3) - A(D<0)/3;
        else
            error('The beta you specified is not currently supported.')
        end

        % update for H_+ and W_+
        Hplus = max(H + 1/rho*alphaH, 0);
        Wplus = max(W + 1/rho*alphaW, 0);
        
        % update for dual variables
        alphaX = alphaX + rho*(X - X_ap);
        alphaH = alphaH + rho*(H - Hplus);
        alphaW = alphaW + rho*(W - Wplus);
        
%         WW = W; WW(:,free) = Wplus(:,free);
%         HH = Hplus;
        
        if KL
            obs = [obs comp_obj(V,Wplus,Hplus)];
        else
            obs = [obs comp_obj_IS(V,Wplus,Hplus)];
        end
        
        times = [times toc-times(1)];
        
        if obs(end) <= eps
            break;
        end
    end
    
    W(:,free) = Wplus(:,free);
    H = Hplus; 

end

function obj = comp_obj(A,W,H)
WH = W*H;
% obj = sum(sum(A.*log(A+1e-10)-A.*log(WH+1e-10)-A+WH));
obj = norm(A-WH)/norm(A);
end

function obj = comp_obj_IS(A,W,H)
[m,n] = size(A);
WH = W*H;
WH = WH+1e-10;
obj = sum(sum((A./WH - log(A./WH)))) - m*n;
end