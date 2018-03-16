%液晶加载阻抗表面 20*20（修正）
%先去掉10_10/10_11/11_10/11_11，再手动挖空各个液晶位置的介质

clc;
clear all;
close all;

tmpPrjFile    = [pwd, '\tmpLC_Metasurface.hfss'];
tmpDataFile   = [pwd, '\tmpData.m'];
tmpScriptFile = [pwd, '\LC_Metasurface_45_45_x.vbs'];

%Parameter(mm)
UnitLength=2.2;
Length_X=44;
Length_Y=44;
m=Length_X/UnitLength;%20
n=Length_Y/UnitLength;%20
PatchLength=2;
GapLength_Half=(UnitLength-PatchLength)/2;
SubHeight=0.127;
Theta=45;%[0,90]
Phi=45;%[0,90)

fid = fopen(tmpScriptFile, 'wt');

hfssNewProject(fid);
hfssInsertDesign(fid, 'LC_Metasurface_45_45_x_171123');

%GND
hfssRectangle(fid, 'GND', 'Z', [-Length_X/2,-Length_Y/2,0], Length_X, Length_Y, 'mm');
hfssAssignPE(fid, 'PE_GND', 0, {'GND'});%平面算作一个object
hfssSetColor(fid, 'GND', [255, 128, 0]);

%Substrate
hfssBox(fid, 'Substrate', [-Length_X/2,-Length_Y/2,0], [Length_X,Length_Y,SubHeight], 'mm');
hfssAssignMaterial(fid, 'Substrate', 'FR4_epoxy');
hfssSetTransparency(fid, {'Substrate'}, 0.25);

%Liquid Crystal dielectric constant
[y,x]=meshgrid(-(Length_Y/2-UnitLength/2)/1000:UnitLength/1000:(Length_Y/2-UnitLength/2)/1000, -(Length_X/2-UnitLength/2)/1000:UnitLength/1000:(Length_X/2-UnitLength/2)/1000);%见文件夹中图片说明
z=zeros(m,n);
lable=zeros(m,n);%修正标识
for i=1:m
    for j=1:n
        if y(i,j)<-x(i,j)/tan(Phi*pi/180)
            z(i,j)=355.6267534+37.989726*cos(35500*pi/99*(x(i,j)^2+y(i,j)^2).^0.5 - 2*pi*130* (x(i,j)*sin(Phi*pi/180)+y(i,j)*cos(Phi*pi/180)) *sin(Theta*pi/180) + pi);%修正
            lable(i,j)=1;%修正标识
        elseif y(i,j)<-x(i,j)/tan(Phi*pi/180) % y<k*x,k=-1/tan(Phi),+pi修正
            z(i,j)=355.6267534+37.989726*cos(35500*pi/99*(x(i,j)^2+y(i,j)^2).^0.5 - 2*pi*130* (x(i,j)*sin(Phi*pi/180)+y(i,j)*cos(Phi*pi/180)) *sin(Theta*pi/180) + pi);%修正
            lable(i,j)=1;%修正标识
        else
            z(i,j)=355.6267534+37.989726*cos(35500*pi/99*(x(i,j)^2+y(i,j)^2).^0.5 - 2*pi*130* (x(i,j)*sin(Phi*pi/180)+y(i,j)*cos(Phi*pi/180)) *sin(Theta*pi/180));%未修正
        end
    end
end

LC_RP=zeros(m,n);
for i=1:m
    for j=1:n
        a=[z(i,j)-1235,5528,-12183,10240];
        b=roots(a);
        LC_RP_temp=zeros(1,3);
        for l=1:3
            if isreal(b(l))
                LC_RP_temp(l)=b(l);
            end
        end
        LC_RP(i,j)=max(LC_RP_temp);
        if LC_RP(i,j)>3.3
            LC_RP(i,j)=3.3;
        end
        if LC_RP(i,j)<2.5
            LC_RP(i,j)=2.5;
        end
    end
end

%Cell of Liquid Crystal & Patch
for i=1:m
    for j=1:n
        i_temp=int2str(i);
        j_temp=int2str(j);
        name_temp=strcat(i_temp,'_');
        name_temp=strcat(name_temp,j_temp);
        LC_name=strcat('LC',name_temp);
        Patch_name=strcat('P',name_temp);
        Material_name=strcat('M',name_temp);
        
        %Liquid Crystal
        hfssBox(fid, LC_name, [-Length_X/2+GapLength_Half+UnitLength*(i-1),-Length_Y/2+GapLength_Half+UnitLength*(j-1),0], [PatchLength,PatchLength,SubHeight], 'mm');
        hfssAddMaterial(fid, Material_name, LC_RP(i,j), 0, 0.0143);%添加介质
        hfssAssignMaterial(fid, LC_name, Material_name);%分配介质
        hfssSetTransparency(fid, {LC_name}, 0.75);
        %hfssSubtract(fid, {'Substrate'}, {LC_name}, 0);%有语法错误
        
        %Patch
        hfssRectangle(fid, Patch_name, 'Z', [-Length_X/2+GapLength_Half+UnitLength*(i-1),-Length_Y/2+GapLength_Half+UnitLength*(j-1),SubHeight], PatchLength,PatchLength, 'mm');
        hfssSetColor(fid, Patch_name, [255, 128, 0]);
    end
end

%AirBox
hfssBox(fid, 'Air', [-Length_X/2-8,-Length_Y/2-8,-8], [Length_X+16,Length_Y+16,SubHeight+16], 'mm');
hfssSetTransparency(fid, {'Air'}, 0.9);

hfssSaveProject(fid, tmpPrjFile, true);
fclose(fid);


