%% Initialization
% Here, the x-axis correspond to coloum, while the y-axis correspond to row
clear all; close all;
img = imread('lenna.jpg');
[~,~,ColorChannel] = size(img);
if ColorChannel > 1
    img = rgb2gray(img);
end
img = double(img)/255;
ScaleSpaceNum = 3; % number of scale space intervals
SigmaOrigin = 2^0.5; % default sigma
ScaleFactor = 2^(1/ScaleSpaceNum);
StackNum = ScaleSpaceNum + 3; % number of stacks = number of scale space intervals + 3
OctaveNum = 3;
GaussianFilterSize = 21;
OctaveImage = {OctaveNum,StackNum}; % save the Gaussian-filtered results of image
OctaveImageDiff = {OctaveNum,StackNum-1}; % save the Difference of Gaussian-filtered results of image


%% Gaussian Convolution of Images in Each Octave
ImgOctave = cell(OctaveNum,1);
for Octave = 1:OctaveNum
    Sigma = SigmaOrigin * 2^(Octave-1); % when up to a new octave, double the sigma
    ImgOctave{Octave} = imresize(img, 2^(-(Octave-1)));
    for s = 1:StackNum
        SigmaScale = Sigma * ScaleFactor^(s-2);
        % calculate Guassian kernel
        GaussianFilter = fspecial('gaussian',[GaussianFilterSize,GaussianFilterSize],SigmaScale);
        % do convolution with Gassian kernel
        OctaveImage{Octave,s} = imfilter(ImgOctave{Octave}, GaussianFilter,'symmetric');
    end  
end
% calculate difference of Gaussian (in original paper eq.1)
for Octave = 1:OctaveNum
    for s = 1:StackNum-1
        OctaveImageDiff{Octave,s} = OctaveImage{Octave,s+1} - OctaveImage{Octave,s};
    end
end
%% Find the local minima and maxima between stacks
DiffMinMaxMap = cell(OctaveNum,StackNum-3);
for Octave = 1:OctaveNum
    for s = 2:size(OctaveImageDiff,2)-1
        CompareDiffImg = zeros(size(OctaveImage{Octave,1},1)-2,size(OctaveImage{Octave,1},2)-2,27);
        indx = 0; % 3rd dimension indx for CompareDiffImg
        for s2 = s-1:s+1
            for k = 1:9
                [i,j] = ind2sub([3,3],k);
                CompareDiffImg(:,:,indx+k) = OctaveImageDiff{Octave,s2}(i:end-3+i,j:end-3+j);
            end
            indx = indx + 9;
        end
        [~,MinMap] = min(CompareDiffImg,[],3);
        MinMap = (MinMap == 14);
        [~,MaxMap] = max(CompareDiffImg,[],3);
        MaxMap = (MaxMap == 14);
        DiffMinMaxMap{Octave,s-1} = MinMap + MaxMap; % the center indx is 9 + 5 = 14
    end
end

%% Accurate keypoint localization
UnstableExtremaThreshold = 0.03; % set the threshold as 0.03 (the same as the original paper)
DiffMinMaxMapAccurate = DiffMinMaxMap;
for Octave = 1:OctaveNum
    Sigma = SigmaOrigin * 2^(Octave-1);
    for DiffMinMaxMapIndx = 1:size(DiffMinMaxMap,2)
        Map = DiffMinMaxMap{Octave,DiffMinMaxMapIndx};
        SSCindx = find(Map); % Scale-Space-Corner Index
        if isempty(SSCindx)
            continue;
        end
        for ssc = 1:length(SSCindx)
            [Row,Col] = ind2sub([size(Map,1),size(Map,2)], SSCindx(ssc));
            if (Row <= 1) || (Row >= size(Map,1)) || (Col <= 1) || (Col >= size(Map,2))
                DiffMinMaxMapAccurate{Octave,DiffMinMaxMapIndx}(Row,Col) = 0; % discard out of matrix boundary
                continue;
            end
            RowDiff = OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row+1,Col) - OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row-1,Col);
            ColDiff = OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row,Col+1) - OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row,Col-1);
            ScaleDiff = OctaveImageDiff{Octave,DiffMinMaxMapIndx+2}(Row,Col) - OctaveImageDiff{Octave,DiffMinMaxMapIndx}(Row,Col);
            offset = [2; 2; Sigma * ScaleFactor^(DiffMinMaxMapIndx) - Sigma * ScaleFactor^(DiffMinMaxMapIndx-2)];
            DxHat = OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row,Col) + 0.5 * ([RowDiff,ColDiff,ScaleDiff] * offset);
            if abs(DxHat) < UnstableExtremaThreshold
                DiffMinMaxMapAccurate{Octave,DiffMinMaxMapIndx}(Row,Col) = 0; % discard unstable extrema
            end
        end
    end
end
%% Eliminating edge responses
gamma = 10; % set the threshold gamma as 10 (the same as the original paper)
DetermineThreshold = (gamma + 1)^2 / gamma;
DiffMinMaxMapNoEdge = DiffMinMaxMapAccurate;
for Octave = 1:OctaveNum
    for DiffMinMaxMapIndx = 1:size(DiffMinMaxMap,2) 
        Map = DiffMinMaxMapAccurate{Octave,DiffMinMaxMapIndx};
        SSCindx = find(Map); % Scale-Space-Corner Index
        if isempty(SSCindx)
            continue;
        end
        for ssc = 1:length(SSCindx)
            [Row,Col] = ind2sub([size(Map,1),size(Map,2)], SSCindx(ssc));
            if (Row <= 1) || (Row >= size(Map,1)) || (Col <= 1) || (Col >= size(Map,2))
                DiffMinMaxMapNoEdge{Octave,DiffMinMaxMapIndx}(Row,Col) = 0; % discard out of matrix boundary
                continue;
            end
            Dyy = OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row+1,Col) - 2*OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row,Col) + OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row-1,Col);
            Dxx = OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row,Col+1) - 2*OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row,Col) + OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row,Col-1);
            Dxy = OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row-1,Col+1) - OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row-1,Col-1) - OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row+1,Col+1) + OctaveImageDiff{Octave,DiffMinMaxMapIndx+1}(Row+1,Col-1);
            TrH = Dxx + Dyy;
            DetH = Dxx*Dyy - Dxy^2;
            if ((TrH^2 / DetH) >= DetermineThreshold)
                DiffMinMaxMapNoEdge{Octave,DiffMinMaxMapIndx}(Row,Col) = 0; % discard unstable extrema
            end
        end
    end
end

%% SIFT feature descriptors generation
% the patch size for dominant orientation calculation is 9;
% the patch size for feature transformation is 16;
DominantOrientation = cell(size(DiffMinMaxMap));
SIFT = cell(size(DiffMinMaxMap));
for Octave = 1:OctaveNum
    Sigma = SigmaOrigin * 2^(Octave-1); % when up to a new octave, double the sigma
    for DiffMinMaxMapIndx = 1:size(DiffMinMaxMap,2)
        stack = DiffMinMaxMapIndx+1;
        SigmaScale = Sigma * ScaleFactor^(stack-2);
        GaussianSmoothedImage = OctaveImage{Octave,stack};
        Map = DiffMinMaxMapNoEdge{Octave,DiffMinMaxMapIndx};
        SSCindx = find(Map); % Scale-Space-Corner Index
        if isempty(SSCindx)
            continue;
        end
        DomOri = zeros(length(SSCindx),2,36); % first column is for sita, second column is for magnitude
        sift = zeros(length(SSCindx),128,36);
        for ssc = 1:length(SSCindx)
            [Row,Col] = ind2sub([size(Map,1),size(Map,2)], SSCindx(ssc));
            Row = Row+1; Col = Col+1; % offset = [1,1];
            if (Row <= 10) || (Row >= size(GaussianSmoothedImage,1)-8) || (Col <= 10) || (Col >= size(GaussianSmoothedImage,2)-8)
                % skip if out of boundary
                continue;
            end
            % magnitude and sita of sample points in the patch
            mag = ((GaussianSmoothedImage(Row-8:Row+7,Col-7:Col+8) - GaussianSmoothedImage(Row-8:Row+7,Col-9:Col+6)).^2 + (GaussianSmoothedImage(Row-7:Row+8,Col-8:Col+7) - GaussianSmoothedImage(Row-9:Row+6,Col-8:Col+7)).^2).^0.5;
            sita = atan2((GaussianSmoothedImage(Row-7:Row+8,Col-8:Col+7) - GaussianSmoothedImage(Row-9:Row+6,Col-8:Col+7)),(GaussianSmoothedImage(Row-8:Row+7,Col-7:Col+8) - GaussianSmoothedImage(Row-8:Row+7,Col-9:Col+6)));
            sita = mod(sita + 2*pi, 2*pi); % the range of atan2 function is -pi~pi, map it to 0~2*pi
            % Dominant orientation calculation
            GaussianFilter = fspecial('gaussian',[9,9],1.5*SigmaScale);
            Dmag = mag(5:13,5:13).* GaussianFilter; % magnitude is weighted by gaussian filter
            sitaquantize = mod(sita(5:13,5:13) + pi/36,2*pi);
            sitaquantize = floor(sitaquantize / (2*pi/36));
            sitabin = zeros(36,1);
            for patchindx = 1:9^2
                sitabin(sitaquantize(patchindx)+1) = sitabin(sitaquantize(patchindx)+1) + Dmag(patchindx);
            end
            maxsitabin = max(sitabin);
            DominantOriBin = find((sitabin / maxsitabin) >= 0.8); % duplicate the feature when non-maximum magnitude of orientation is also big
            DominantOriSita = (DominantOriBin-1)*(2*pi/36);
            DomOri(ssc,1,1:length(DominantOriSita)) = DominantOriSita;
            DomOri(ssc,2,1:length(DominantOriSita)) = sitabin((sitabin / maxsitabin) >= 0.8);
            % SIFT feature calculation
            for DomOriIndx = 1:length(DominantOriSita)
                GaussianFilter = fspecial('gaussian',[16,16],8);
                Smag = mag.* GaussianFilter;
                sitaquantize = mod(sita - DominantOriSita(DomOriIndx) + pi/8 + 2*pi,2*pi);
                sitaquantize = floor(sitaquantize / (2*pi/8));
                sitabin = zeros(8,4,4);
                for patchindx = 1:16^2
                    [row,col] = ind2sub([16,16], patchindx);
                    row = floor((row-1)/4)+1;
                    col = floor((col-1)/4)+1;
                    sitabin(sitaquantize(patchindx)+1,row,col) = sitabin(sitaquantize(patchindx)+1,row,col) + Smag(patchindx);
                end
                sitabin = bsxfun(@times,sitabin,sum(sitabin.^2,1).^(-0.5)); % normalize the vector
                sitabin(sitabin > 0.2) = 0.2; % threshold the maximum value as 0.2
                sitabin = bsxfun(@times,sitabin,sum(sitabin.^2,1).^(-0.5)); % renormalize
                sift(ssc,:,DomOriIndx) = sitabin(:);
            end
        end
        DominantOrientation{Octave,DiffMinMaxMapIndx} = DomOri;
        SIFT{Octave,DiffMinMaxMapIndx} = sift;
    end 
end

%% Show results (this part of code is just for verification, and is a mess...)
% close all;
% imgtemp = img*0.5;
% imgtemp(2:end-1,2:end-1) = (imgtemp(2:end-1,2:end-1) + (DiffMinMaxMap{1,1}+DiffMinMaxMap{1,2}+DiffMinMaxMap{1,3}))*255;
% figure,imshow(uint8(imgtemp));
% imgtemp2 = img*0.5;
% imgtemp2(2:end-1,2:end-1) = (imgtemp2(2:end-1,2:end-1) + (DiffMinMaxMapAccurate{1,1}+DiffMinMaxMapAccurate{1,2}+DiffMinMaxMapAccurate{1,3}))*255;
% figure,imshow(uint8(imgtemp2));
% imgtemp3 = img*0.5;
% imgtemp3(2:end-1,2:end-1) = (imgtemp3(2:end-1,2:end-1) + (DiffMinMaxMapNoEdge{1,1}+DiffMinMaxMapNoEdge{1,2}+DiffMinMaxMapNoEdge{1,3}))*255;
% figure,imshow(uint8(imgtemp3));
imgtemp4 = img*0.5;
figure,imshow(uint8(imgtemp4*255));
for i = 1:3
for j = 1:3
for k = 1:2
if isempty(DominantOrientation{i,j})
continue
end
DomOri = DominantOrientation{i,j}(:,:,k);
DOx = cos(DomOri(:,1)).*DomOri(:,2)*200;
DOy = sin(DomOri(:,1)).*DomOri(:,2)*200;
Map = DiffMinMaxMapNoEdge{i,j};
SSCindx = find(Map); % Scale-Space-Corner Index
for ssc = 1:length(SSCindx)
    hold on;
    [Row,Col] = ind2sub([size(Map,1),size(Map,2)], SSCindx(ssc));
    Row = Row*2^(i-1)+1; Col = Col*2^(i-1)+1; % offset = [1,1];
    if (Row > size(img,1)) || (Col > size(img,2))
        continue
    end
    quiver(Col,Row,DOx(ssc),DOy(ssc),'r')
end
end
end
end
hold off;
%
imgtemp5 = img*0.5;
for i = 1:3
for j = 1:3
for k = 1:2
if isempty(DominantOrientation{i,j})
continue
end
Map = DiffMinMaxMapNoEdge{i,j};
SSCindx = find(Map); % Scale-Space-Corner Index
for ssc = 1:length(SSCindx)
    [Row,Col] = ind2sub([size(Map,1),size(Map,2)], SSCindx(ssc));
    Row = Row*2^(i-1)+1; Col = Col*2^(i-1)+1; % offset = [1,1];
    if (Row > size(img,1)) || (Col > size(img,2))
        continue
    end
    imgtemp5(Row,Col) = 1;
end
end
end
end
figure,imshow(uint8(imgtemp5*255));
