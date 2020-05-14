clear;clc;
% Tempatkan model data anda pada file direktori yang diinginkan lalu ambil keseluruhan model data
% Place your data model to directory file you want and get all data model
files_kelana=dir('C:\Users\User\Documents\MATLAB\SIFT\Data Image\Topeng Kelana\*.png');
files_tumenggung=dir('C:\Users\User\Documents\MATLAB\SIFT\Data Image\Topeng Tumenggung\*.png');
files_rumyang=dir('C:\Users\User\Documents\MATLAB\SIFT\Data Image\Topeng Rumyang\*.png');
files_samba=dir('C:\Users\User\Documents\MATLAB\SIFT\Data Image\Topeng Samba\*.png');
files_panji=dir('C:\Users\User\Documents\MATLAB\SIFT\Data Image\Topeng Panji\*.png');
n_kelana =numel(files_kelana);
n_tumenggung =numel(files_tumenggung);
n_rumyang =numel(files_rumyang);
n_samba =numel(files_samba);
n_panji =numel(files_panji);
class_kelana=1;
class_tumenggung=2;
class_rumyang=3;
class_samba=4;
class_panji=5;
circle = [32 32 32];

%fitur kelana
%Klana feature with SIFT
for i=1:n_kelana
      str = strcat('C:\Users\User\Documents\MATLAB\SIFT\Data Image\Topeng Kelana\', files_kelana(i).name);
      temp_image=imread(str);
      feature_sift_kelana(i,:) = find_sift(temp_image, circle);
      feature_class_kelana(i,:)=[feature_sift_kelana(i,:) class_kelana];
      
end

%fitur tumenggung
%Tumenggung Feature
for i=1:n_tumenggung
    str = strcat('C:\Users\User\Documents\MATLAB\SIFT\Data Image\Topeng Tumenggung\', files_tumenggung(i).name);
      temp_image=imread(str);
      feature_sift_tumenggung(i,:) = find_sift(temp_image, circle);
      feature_class_tumenggung(i,:)=[feature_sift_tumenggung(i,:) class_tumenggung];

end

%fitur rumyang
%Rumyang Feature
for i=1:n_rumyang
    str = strcat('C:\Users\User\Documents\MATLAB\SIFT\Data Image\Topeng Rumyang\', files_rumyang(i).name);
    temp_image=imread(str);
      feature_sift_rumyang(i,:) = find_sift(temp_image, circle);
      feature_class_rumyang(i,:)=[feature_sift_rumyang(i,:) class_rumyang];
      
end

%fitur samba
%Pamindo or Samba Feature
for i=1:n_samba
    str = strcat('C:\Users\User\Documents\MATLAB\SIFT\Data Image\Topeng Samba\', files_samba(i).name);
    temp_image=imread(str);
      feature_sift_samba(i,:) = find_sift(temp_image, circle);
      feature_class_samba(i,:)=[feature_sift_samba(i,:) class_samba];
      
end

%fitur panji
%Panji Feature
for i=1:n_panji
    str = strcat('C:\Users\User\Documents\MATLAB\SIFT\Data Image\Topeng Panji\', files_panji(i).name);
    temp_image=imread(str);
      feature_sift_panji(i,:) = find_sift(temp_image, circle);
      feature_class_panji(i,:)=[feature_sift_panji(i,:) class_panji];
      
end

%Feature all data with SIFT and save result feature data 
feature_all = [feature_class_kelana; feature_class_tumenggung; feature_class_rumyang; feature_class_samba; feature_class_panji];
save('feature.mat', 'feature_all');
