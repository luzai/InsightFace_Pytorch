
cd /home/zl/prj/ 

for m in fcpth.* 
do 
    echo $m
    cp fcpth/Learner.py $m/
    cp fcpth/models/model.py $m/models/
    cp fcpth/train.py $m/
    cp fcpth/install.py $m/ 
    cp fcpth/lz.py $m/
done

for m in fcpth
do 
echo $m 
python3 common_zone_uploader.py --local_folder_absolute_path=/home/zl/prj/$m   --region=cn-north-4 --bucket_name=bucket-2243 --bucket_path=zl/ --exclude=work_space-apex.build-pretrain
done

for m in fcpth
do 
echo $m 
python3 common_zone_uploader.py --local_folder_absolute_path=/home/zl/prj/$m   --region=cn-north-1 --bucket_name=bucket-6944  --bucket_path=zl/ --exclude=work_space-apex.build-pretrain  
done



