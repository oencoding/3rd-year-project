function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
	%简单的说mod(a,b)就是求的是a除以b的余数。比方说mod(100,3)=1,mod(17,6)=5
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end

