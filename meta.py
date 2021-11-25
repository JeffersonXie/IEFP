
FGNET = {
    'file_root': './face_dataset/ID_processed_FGNET',
    'pat': r'A|\.|a|b',
    'pos': 1,
    'n_cls': 82
}


MORPH2_probe_3000= {
    'file_root': './face_dataset/ID_processed_morph2_2/probe_3000',
    'pat': r'M|F|\.',
    'pos': 1,
    'n_cls': 3000
}


MORPH2_gallery_3000 = {
    'file_root': './face_dataset/ID_processed_morph2_2/gallery_3000',
    'pat': r'M|F|\.',
    'pos': 1,
    'n_cls': 3000
}


MORPH2_probe_10000 = {
    'file_root': './face_dataset/ID_processed_morph2_2/probe_10000',
    'pat': r'M|F|\.',
    'pos': 1,
    'n_cls': 10000
}


MORPH2_gallery_10000 = {
    'file_root': './face_dataset/ID_processed_morph2_2/gallery_10000',
    'pat': r'M|F|\.',
    'pos': 1,
    'n_cls': 10000
}


processed_CACD_VS = {
    'file_root': './face_dataset/CACD_VS/processed_CACDVS_2',
    'pair_lists_root': './face_dataset/CACD_VS/txt/CACD_VS_pairs_labels.txt'   
}


processed_CALFW = {
    'file_root': './face_dataset/CALFW/processed_CALFW_2',
    'pair_lists_root': './face_dataset/CALFW/txts/CALFW_img_pairs_labels.txt'
}


ms1mv3 = {
    'file_root': './face_dataset/ms1mv3_256times256_with_age',
    'pat': r'_|\.',
    'pos': 2,
    'n_cls': 93431
}


age_cutoffs = [12, 18, 25, 35, 45, 55, 65]