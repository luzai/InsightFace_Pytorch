set -x
set -e 

# bash extract_dis [model_name] [model_epoch] [gpu_num]

# MODEL_NAME=$1
# MODEL='../models/'$MODEL_NAME'/'$MODEL_NAME
# EPOCH=$2
OUTPUT_ROOT='../output'
# echo $MODEL','$EPOCH

echo $OUTPUT_ROOT
# common test
# python embeddings_test.py --prefix /mnt/109_ssd/ssd1/hzf_data/testsets_mtcnn_align112x112/ --gpu_num $3 --model $MODEL','$EPOCH /mnt/109_ssd/ssd1/hzf_data/testsets_mtcnn_align112x112/jk_all_list_cut.txt "$OUTPUT_ROOT"
# python embeddings_test.py --prefix /mnt/109_ssd/ssd1/hzf_data/testsets_mtcnn_align112x112/ --gpu_num $3 --model $MODEL','$EPOCH /mnt/109_ssd/ssd1/hzf_data/testsets_mtcnn_align112x112/zj_list_cut.txt "$OUTPUT_ROOT"
# MODEL_NAME=$1
# echo $MODEL_NAME

# MODEL_P='/home/zl/prj/InsightFace_Pytorch/work_space/emore.r100.bs.ft.tri.dop.cont/save/,0'
# MODEL_P='/home/zl/prj/InsightFace_Pytorch/work_space/emore.r100.dop.tri.stable.cont/models/,0'
USE_TORCH=0
MODEL_P='/home/zl/prj/insightface/logs/model.emore.jk.test/model,181'
# MODEL_P='/home/zl/prj/models/r100_loss4_mxnet/r100_loss4_mxnet,13',
# MODEL_P='/home/zl/prj/models/r100_se_base+mhyset_0602-0724_ft_bninit_pk/r100_se_base+mhyset_0602-0724_ft_bninit_pk,45', 
# MODEL_NAME='MS1MV2-ResNet100-Arcface'
MODEL_NAME='model'
# MODEL_NAME='emore.r100.bs.ft.tri.dop.cont'

source activate py36 
python embeddings_test.py --images_list ../data/lists/MediaCapturedImagesStd_02_en_sim/jk_all_list.txt --use_torch $USE_TORCH --model $MODEL_P
python embeddings_test.py --images_list ../data/lists/MediaCapturedImagesStd_02_en_sim/zj_list.txt --use_torch $USE_TORCH --model $MODEL_P
python embeddings_test.py --images_list ../data/lists/facereg1N_Select30_sim/jk_all_list.txt --use_torch $USE_TORCH --model $MODEL_P 
python embeddings_test.py --images_list ../data/lists/facereg1N_Select30_sim/zj_list.txt --use_torch $USE_TORCH --model $MODEL_P
# exit

source activate base 
python embeddings_dis.py --use_torch $USE_TORCH --model $MODEL_P
# exit

source activate py36
CUDA_VISIBLE_DEVICES="4,5,6,7" python tf_test_with_log_mgpu.py "$MODEL_NAME"

mkdir -p ../calc_result/$MODEL_NAME
# cp $OUTPUT_ROOT/$MODEL_NAME/testsets_mtcnn_align112x112.log ../calc_result/$MODEL_NAME/hy_set.log
cp $OUTPUT_ROOT/$MODEL_NAME/MediaCapturedImagesStd_02_en_sim.log ../calc_result/$MODEL_NAME/media_en_sim.log
cp $OUTPUT_ROOT/$MODEL_NAME/facereg1N_Select30_sim.log ../calc_result/$MODEL_NAME/facereg1N_Select30_sim.log
# @deprecated
#cp $OUTPUT_ROOT/$MODEL_NAME/hy.log ../calc_result/$MODEL_NAME/hy_set.log
cd ../calc_result

python get_e2e_result.py --input-root $MODEL_NAME

# python embeddings_test.py --prefix ../data/zj_and_jk/ --gpu_num $3 --model $MODEL','$EPOCH ../data/lists/MediaCapturedImagesStd_02_en_sim/jk_all_list.txt "$OUTPUT_ROOT"
# python embeddings_test.py --prefix ../data/zj_and_jk/ --gpu_num $3 --model $MODEL','$EPOCH ../data/lists/MediaCapturedImagesStd_02_en_sim/zj_list.txt "$OUTPUT_ROOT"
# python embeddings_test.py --prefix ../data/zj_and_jk/ --gpu_num $3 --model $MODEL','$EPOCH ../data/lists/facereg1N_Select30_sim/jk_all_list.txt "$OUTPUT_ROOT"
# python embeddings_test.py --prefix ../data/zj_and_jk/ --gpu_num $3 --model $MODEL','$EPOCH ../data/lists/facereg1N_Select30_sim/zj_list.txt "$OUTPUT_ROOT"
