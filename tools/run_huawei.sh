set -x
# set -e
set -v

# bash extract_dis [model_p] [model_name]

OUTPUT_ROOT='../output'
# echo $MODEL','$EPOCH

MODEL_P=$1
# MODEL_P='/home/zl/prj/InsightFace_Pytorch/work_space/alpha.tri.dop.ft/models/,0' '/home/zl/prj/models/MS1MV2-ResNet100-Arcface/MS1MV2-ResNet100-Arcface,0'
USE_TORCH=1
MODEL_NAME=$2
# MODEL_NAME='MS1MV2-ResNet100-Arcface' 'alpha.tri.dop.ft'
echo $OUTPUT_ROOT, $MODEL_P, $MODEL_NAME

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
