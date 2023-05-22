export _BASE_PATH='../omniglot_resized'
export _NUM_PRETRAIN_CLASSES=1200
export _NUM_FINETUNE_CLASSES=423
export _NUM_SAMPLES_PER_CLASS=20
export _SPLIT_PATH='../omniglot_split'
export _PRETRAIN_FOLDER_NAME='pretrain'
export _FINETUNE_FOLDER_NAME='finetune'

clearenv () {
    unset _BASE_PATH
    unset _NUM_PRETRAIN_CLASSES
    unset _NUM_FINETUNE_CLASSES
    unset _NUM_SAMPLES_PER_CLASS
    unset _SPLIT_PATH
    unset _PRETRAIN_FOLDER_NAME
    unset _FINETUNE_FOLDER_NAME
}
