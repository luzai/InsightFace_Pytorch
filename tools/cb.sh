#!/usr/bin/env bash
set -x
set -e

export OUTPUT_ROOT='/home/zl/prj/output'
export PREFIX=/home/zl/prj/
USE_TORCH=1
# DATA_PREFIX=/home/zl/prj/data.new/
# DATA_PREFIX=/ssd/ssd0/z84115054/data.new/
#DATA_PREFIX=/home/zl/prj/data_old/
DATA_PREFIX=/ssd/ssd0/z84115054/data_old/

# MODEL_NAME=emore.r100.dop.alphaf64.chsfst
# MODEL_NAME=alpha.sc.idsmpl # this is r50 model 
# MODEL_NAME=alphaf64.dop.tri.chsfst.ft.4
MODEL_NAME=$1
# MODEL_NAME=r50.alpha.elu.in.ranger
# MODEL_NAME=alpha.neg.25.15.1e-1.top5.ft.2
# MODEL_NAME=alpha.neg.25.15.top5.ft.2
# MODEL_NAME='MS1MV2-ResNet100-Arcface'
# MODEL_NAME='model'

CODE_P=$2
# CODE_P="$PREFIX/fcpth.r100.ranger" 
# CODE_P="$PREFIX/fcpth.r100.bs" 
# CODE_P="$PREFIX/arc/fcpth.noipabn" 

# MODEL_P= "$PREFIX/work_space/alpha.neg.25.15.1e-1.top5.ft/models/model_2019-09-07-17_accuracy:0.9661428571428571_step:436392_final.pth,0" # tested
# MODEL_P= '$PREFIX/work_space/alpha.neg.25.15.top5.ft/models/model_2019-09-08-03_accuracy:0.9638571428571427_step:436392_final.pth,0' # tested
# MODEL_P= '$PREFIX/work_space/alpha.neg.15.2.top5.ft/models/model_2019-09-07-19_accuracy:0.9632857142857141_step:436392_final.pth,0'
# MODEL_P= '$PREFIX/work_space/alpha.neg.2.2.top3.ft/models/model_2019-09-03-21_accuracy:0.9628571428571429_step:436392_final.pth,0'
# MODEL_P= '$PREFIX/work_space/r100.alpha.neg.25.1.top3/models/model_2019-09-29-09_accuracy:0.9427142857142858_step:88740_None.pth,0'
# MODEL_P= '$PREFIX/work_space/alpha.neg.25.15.top5.ft/models/model_2019-09-06-19_accuracy:0.9392857142857143_step:327294_final.pth,0'
#  '$PREFIX/work_space/r100.neg.3.1.top3/models/model_2019-09-29-09_accuracy:0.9229999999999998_step:53244_None.pth', # not intend to test 
#  '$PREFIX/work_space/r100.neg.48.02.top3/models/model_2019-09-29-08_accuracy:0.8922857142857143_step:35496_None.pth',
# MODEL_P="$CODE_P/work_space/$MODEL_NAME/models/,0"
MODEL_P="$PREFIX/work_space/$MODEL_NAME/models/,0" 
# MODEL_P='$PREFIX/insightface/logs/model.emore.jk.test/model,181'
# MODEL_P='$PREFIX/models/r100_loss4_mxnet/r100_loss4_mxnet,13',
# MODEL_P='$PREFIX/models/r100_se_base+mhyset_0602-0724_ft_bninit_pk/r100_se_base+mhyset_0602-0724_ft_bninit_pk,45', 
# MODEL_P="$PREFIX/models/$MODEL_NAME/$MODEL_NAME,0"
echo $MODEL_P $MODEL_NAME

python embeddings_test.py --code $CODE_P --images_list $DATA_PREFIX/lists/MediaCapturedImagesStd_02_en_sim/jk_all_list.txt --use_torch $USE_TORCH --model $MODEL_P --prefix $DATA_PREFIX/zj_and_jk/ --gpu_num 4 --model_name $MODEL_NAME &
python embeddings_test.py --code $CODE_P --images_list $DATA_PREFIX/lists/MediaCapturedImagesStd_02_en_sim/zj_list.txt --use_torch $USE_TORCH --model $MODEL_P --prefix $DATA_PREFIX/zj_and_jk/ --gpu_num 5 --model_name $MODEL_NAME &
python embeddings_test.py --code $CODE_P --images_list $DATA_PREFIX/lists/facereg1N_Select30_sim/jk_all_list.txt --use_torch $USE_TORCH --model $MODEL_P --prefix $DATA_PREFIX/zj_and_jk/ --gpu_num 6 --model_name $MODEL_NAME &
python embeddings_test.py --code $CODE_P --images_list $DATA_PREFIX/lists/facereg1N_Select30_sim/zj_list.txt --use_torch $USE_TORCH --model $MODEL_P --prefix $DATA_PREFIX/zj_and_jk/ --gpu_num 7 --model_name $MODEL_NAME

CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7" python embeddings_dis.py --code $CODE_P --use_torch $USE_TORCH --model $MODEL_P --prefix $DATA_PREFIX --images_list $DATA_PREFIX/lists/dis/dis_list.txt --batch_size 256 --model_name $MODEL_NAME

CUDA_VISIBLE_DEVICES="4,5,6,7" python tf_test_with_log_mgpu.py "$MODEL_NAME" --prefix $DATA_PREFIX 

mkdir -p $PREFIX/calc_result/$MODEL_NAME
cp $OUTPUT_ROOT/$MODEL_NAME/MediaCapturedImagesStd_02_en_sim.log $PREFIX/calc_result/$MODEL_NAME/media_en_sim.log
cp $OUTPUT_ROOT/$MODEL_NAME/facereg1N_Select30_sim.log $PREFIX/calc_result/$MODEL_NAME/facereg1N_Select30_sim.log
cd $PREFIX/calc_result

python get_e2e_result.py --input-root $MODEL_NAME

