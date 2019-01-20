#MODEL='/data/109_hzf_home/data/workspace/reproduce_mrc/insightface/models/mrc_dataset_multifc_model/mrc_dataset_multifc_model'
#MODEL='/data/109_hzf_home/data/workspace/face_recognition/models/resnet-se-l-100-mrc-mspek/resnet-se-l-100-mrc-mspek'
#MODEL='/data/109_hzf_home/data/workspace/reproduce_mrc/insightface/models/facesemore+peking_dataset_model_r100/faceemore+peking_dataset_model_r100'
#MODEL='/data/109_hzf_home/data/workspace/reproduce_mrc/insightface/models/mtcnn_align_mspk_r100/mtcnn_align_mspk_r100'
# bash extract_dis [model_name] [model_epoch] [gpu_num]
MODEL_NAME=$1
MODEL='../models/'$MODEL_NAME'/'$MODEL_NAME
EPOCH=$2
OUTPUT_ROOT='../output'
echo $MODEL','$EPOCH
echo $OUTPUT_ROOT
python embeddings_dis.py --prefix ../data/ --gpu_num $3 --model $MODEL','$EPOCH ../data/lists/dis/dis_list.txt "$OUTPUT_ROOT"
# python new_test_with_log.py "$MODEL_NAME"
CUDA_VISIBLE_DEVICES="$3" python tf_test_with_log_mgpu.py "$MODEL_NAME"
mkdir ../calc_result/$MODEL_NAME
cp $OUTPUT_ROOT/$MODEL_NAME/testsets_mtcnn_align112x112.log ../calc_result/$MODEL_NAME/hy_set.log
cp $OUTPUT_ROOT/$MODEL_NAME/MediaCapturedImagesStd_02_en_sim.log ../calc_result/$MODEL_NAME/media_en_sim.log
cp $OUTPUT_ROOT/$MODEL_NAME/facereg1N_Select30_sim.log ../calc_result/$MODEL_NAME/facereg1N_Select30_sim.log
# @deprecated
#cp $OUTPUT_ROOT/$MODEL_NAME/hy.log ../calc_result/$MODEL_NAME/hy_set.log
cd ../calc_result
python get_e2e_result.py --input-root $MODEL_NAME
# python calc_intersection.py --input-root $MODEL_NAME
cd -
