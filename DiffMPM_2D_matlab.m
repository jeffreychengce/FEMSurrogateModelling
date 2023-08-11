numelem = 320;

pCount = numelem; % # of particles
Mp = ones(pCount, 1); % mass
Vp = ones(pCount, 1); % volume
Fp = ones(pCount, 4); 
s = zeros(pCount, 3); % gradient deformation
eps = zeros(pCount, 3); % stress
vp = zeros(pCount, 2); % strain
xp = zeros(pCount, 2); 

for e = 1:numelem
    coord = node1(element1(e,:),:);
    a = det([coord,[1;1;1]])/2;
    Vp(e) = a;
    Mp(e) = 
end
