import os
from .coco_detection import CocoDetection
from .nus_wide import NUSWIDE_ZSL
from .pascal_voc import voc2007
from .nus_wide_text_image import NUSWIDE_ZSL_TEXT_IMAGE
from .coco_detection_text_image import CocoDetection_TEXT_IMAGE

MODEL_TABLE = {
    'coco': CocoDetection,
    'nus_wide_zsl': NUSWIDE_ZSL,
    'voc2007': voc2007,
    'nus_wide_zsl_text_image': NUSWIDE_ZSL_TEXT_IMAGE,
    "coco_text_image": CocoDetection_TEXT_IMAGE
}



def build_dataset(cfg, data_split, annFile=""):
    print(' -------------------- Building Dataset ----------------------')
    print('DATASET.ROOT = %s' % cfg.DATASET.ROOT)
    print('data_split = %s' % data_split)
    print('PARTIAL_PORTION= %f' % cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION)
    if annFile != "":
        annFile = os.path.join(cfg.DATASET.ROOT, 'annotations', annFile)
    try:
        if 'train' in data_split or 'Train' in data_split:
            img_size = cfg.INPUT.TRAIN.SIZE[0]
        else:
            img_size = cfg.INPUT.TEST.SIZE[0]
    except:
        img_size = cfg.INPUT.SIZE[0]
    print('INPUT.SIZE = %d' % img_size)
    
    if cfg.DATASET.NAME == 'nus_wide_zsl_text' and data_split != 'train':
        return MODEL_TABLE[cfg.DATASET.NAME[:-5]](cfg.DATASET.ROOT, data_split, img_size,
                                    p=cfg.DATALOADER.TRAIN_X.PORTION, annFile=annFile,
                                    label_mask=cfg.DATASET.MASK_FILE,
                                    partial=cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION)
    elif cfg.DATASET.NAME == 'nus_wide_zsl_text_image' and data_split != 'train':
        return MODEL_TABLE['nus_wide_zsl'](cfg.DATASET.ROOT, data_split, img_size,
                            p=cfg.DATALOADER.TRAIN_X.PORTION, annFile=annFile,
                            label_mask=cfg.DATASET.MASK_FILE, 
                            partial=cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION)
    else: 
        if 'train' in data_split:
            dataset_name = cfg.DATASET.NAME + '_text_image'
            return MODEL_TABLE[dataset_name](cfg.DATASET.ROOT, data_split, img_size,
                                p=cfg.DATALOADER.TRAIN_X.PORTION, annFile=annFile,
                                label_mask=cfg.DATASET.MASK_FILE,
                                partial=cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION)
        else:
            return MODEL_TABLE[cfg.DATASET.NAME](cfg.DATASET.ROOT, data_split, img_size,
                                            p=cfg.DATALOADER.TRAIN_X.PORTION, annFile=annFile,
                                            label_mask=cfg.DATASET.MASK_FILE,
                                            partial=cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION)

