function [  theta_M  ,  r_M ]= Beam_Squint_trajectory(B, M, f  ,   theta0, r0,  thetac,  rc )

theta_M =zeros(M,1);
r_M = zeros(M,1);
for m = 1 : M
    theta_M(m) = asin(     (B-(f(m)-f(1)))*f(1)/B/f(m)*sin( theta0)    + (B + f(1))*(f(m)-f(1))/B/f(m)*sin(thetac )       );
    r_M(m) = 1/(      1/r0* (B-(f(m)-f(1)))*f(1)/B/f(m)*cos(theta0)* cos(theta0) /cos(theta_M(m))/cos(theta_M(m))   + 1 / rc *  (B + f(1))*(f(m)-f(1))/B/f(m)*   cos(thetac) * cos(thetac) /   cos(theta_M(m))/cos(theta_M(m))           );
end



end

