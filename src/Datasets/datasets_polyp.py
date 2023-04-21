import os

def dataset_creation(show=True):
    
    # Image Paths For Train Dataset
    # Train Dataset contains 900 Images from Kvasir Dataset & 550 Images from CVC-ClinicDB Dataset
    PATH_TO_IMAGES_TRAIN = './Datasets/TrainDataset_Polyp/TrainDataset/images'
    PATH_TO_IMAGES_TRAIN_GT_Object = './Datasets/TrainDataset_Polyp/TrainDataset/masks'

    # Image Paths For Test From CVC-300 Dataset
    PATH_TO_IMAGES_TEST_CVC_300 = './Datasets/TestDataset_Polyp/CVC-300/images'
    PATH_TO_IMAGES_TEST_CVC_300_GT_Object = './Datasets/TestDataset_Polyp/CVC-300/masks'

    # Image Paths For Test From CVC-ClinicDB Dataset
    PATH_TO_IMAGES_TEST_CVC_CLINICDB = './Datasets/TestDataset_Polyp/CVC-ClinicDB/images'
    PATH_TO_IMAGES_TEST_CVC_CLINICDB_GT_Object = './Datasets/TestDataset_Polyp/CVC-ClinicDB/masks'

    # Image Paths For Test From CVC-ColonDB Dataset
    PATH_TO_IMAGES_TEST_CVC_COLONDB = './Datasets/TestDataset_Polyp/CVC-ColonDB/images'
    PATH_TO_IMAGES_TEST_CVC_COLONDB_GT_Object = './Datasets/TestDataset_Polyp/CVC-ColonDB/masks'

    # Image Paths For Test From ETIS Dataset
    PATH_TO_IMAGES_TEST_ETIS = './Datasets/TestDataset_Polyp/ETIS-LaribPolypDB/images'
    PATH_TO_IMAGES_TEST_ETIS_GT_Object = './Datasets/TestDataset_Polyp/ETIS-LaribPolypDB/masks'

    # Image Paths For Test From Kvasir Dataset
    PATH_TO_IMAGES_TEST_KVASIR = './Datasets/TestDataset_Polyp/Kvasir/images'
    PATH_TO_IMAGES_TEST_KVASIR_GT_Object = './Datasets/TestDataset_Polyp/Kvasir/masks'

    # define train variables from Train Dataset
    train_images_half_path = sorted(os.listdir(PATH_TO_IMAGES_TRAIN))
    train_images_half_path_gt_object = sorted(os.listdir(PATH_TO_IMAGES_TRAIN_GT_Object))

    # define test variables from CVC-300 Dataset
    test_images_cvc_300_half_path = sorted(os.listdir(PATH_TO_IMAGES_TEST_CVC_300))
    test_images_cvc_300_half_path_gt_object = sorted(os.listdir(PATH_TO_IMAGES_TEST_CVC_300_GT_Object))

    # define test variables from CVC-ClinicDB Dataset
    test_images_cvc_clinicdb_half_path = sorted(os.listdir(PATH_TO_IMAGES_TEST_CVC_CLINICDB))
    test_images_cvc_clinicdb_half_path_gt_object = sorted(os.listdir(PATH_TO_IMAGES_TEST_CVC_CLINICDB_GT_Object))

    # define test variables from CVC-ColonDB Dataset
    test_images_cvc_colondb_half_path = sorted(os.listdir(PATH_TO_IMAGES_TEST_CVC_COLONDB))
    test_images_cvc_colondb_half_path_gt_object = sorted(os.listdir(PATH_TO_IMAGES_TEST_CVC_COLONDB_GT_Object))

    # define test variables from ETIS Dataset
    test_images_etis_half_path = sorted(os.listdir(PATH_TO_IMAGES_TEST_ETIS))
    test_images_etis_half_path_gt_object = sorted(os.listdir(PATH_TO_IMAGES_TEST_ETIS_GT_Object))

    # define test variables from Kvasir Dataset
    test_images_kvasir_half_path = sorted(os.listdir(PATH_TO_IMAGES_TEST_KVASIR))
    test_images_kvasir_half_path_gt_object = sorted(os.listdir(PATH_TO_IMAGES_TEST_KVASIR_GT_Object))

    # Train Dataset
    train_images = []
    train_images_gt_object = []

    # Test Dataset
    # CVC-300
    test_images_cvc_300 = []
    test_images_cvc_300_gt_object = []

    # CVC-ClinicDB 
    test_images_cvc_clinicdb = []
    test_images_cvc_clinicdb_gt_object = []

    # CVC-ColonDB
    test_images_cvc_colondb = []
    test_images_cvc_colondb_gt_object = []
    
    # ETIS 
    test_images_etis = []
    test_images_etis_gt_object = []

    # Kvasir
    test_images_kvasir = []
    test_images_kvasir_gt_object = []

    # insert the full path to train images from Train Dataset
    for i in range(len(train_images_half_path)):
        train_images.append(PATH_TO_IMAGES_TRAIN + "/{}".format(train_images_half_path[i]))
        train_images_gt_object.append(PATH_TO_IMAGES_TRAIN_GT_Object + "/{}".format(train_images_half_path_gt_object[i]))

    # insert the full path to each test image from CVC-300 Dataset
    for i in range(len(test_images_cvc_300_half_path)):
        test_images_cvc_300.append(PATH_TO_IMAGES_TEST_CVC_300 + "/{}".format(test_images_cvc_300_half_path[i]))
        test_images_cvc_300_gt_object.append(PATH_TO_IMAGES_TEST_CVC_300_GT_Object + "/{}".format(test_images_cvc_300_half_path_gt_object[i]))
                
    # insert the full path to each test image from CVC-ClinicDB  Dataset        
    for i in range(len(test_images_cvc_clinicdb_half_path)):
        test_images_cvc_clinicdb.append(PATH_TO_IMAGES_TEST_CVC_CLINICDB + "/{}".format(test_images_cvc_clinicdb_half_path[i]))
        test_images_cvc_clinicdb_gt_object.append(PATH_TO_IMAGES_TEST_CVC_CLINICDB_GT_Object + "/{}".format(test_images_cvc_clinicdb_half_path_gt_object[i]))
    
    # insert the full path to each test image from CVC-ColonDB Dataset
    for i in range(len(test_images_cvc_colondb_half_path)):
        test_images_cvc_colondb.append(PATH_TO_IMAGES_TEST_CVC_COLONDB + "/{}".format(test_images_cvc_colondb_half_path[i]))
        test_images_cvc_colondb_gt_object.append(PATH_TO_IMAGES_TEST_CVC_COLONDB_GT_Object + "/{}".format(test_images_cvc_colondb_half_path_gt_object[i]))
    
    # insert the full path to each train-test image from ETIS Dataset
    for i in range(len(test_images_etis_half_path)):
        test_images_etis.append(PATH_TO_IMAGES_TEST_ETIS + "/{}".format(test_images_etis_half_path[i]))
        test_images_etis_gt_object.append(PATH_TO_IMAGES_TEST_ETIS_GT_Object + "/{}".format(test_images_etis_half_path_gt_object[i]))

    # insert the full path to each train-test image from Kvasir Dataset
    for i in range(len(test_images_kvasir_half_path)):
        test_images_kvasir.append(PATH_TO_IMAGES_TEST_KVASIR + "/{}".format(test_images_kvasir_half_path[i]))
        test_images_kvasir_gt_object.append(PATH_TO_IMAGES_TEST_KVASIR_GT_Object + "/{}".format(test_images_kvasir_half_path_gt_object[i]))
    
    if show:
        print('[INFO] Initializing Train Dataset composed by Kvasir +  CVC-ClinicDB')
        print(60*'-')
        print(10*' ' + f'Train Images From Train Dataset are => {len(train_images)}')
        print(10*' ' + f'Train Images With GT-Object are => {len(train_images_gt_object)}')
        print(60*'-')
        print('[INFO] Initializing Test Datasets composed by CVC-300, CVC-ClinicDB, CVC-ColonDB, ETIS and KVASIR')
        print(60*'-')
        print(10*' ' + f'Test Images From CVC-300 Dataset are => {len(test_images_cvc_300)}')
        print(10*' ' + f'Test Images With GT-Object From CVC-300 Dataset are => {len(test_images_cvc_300_gt_object)}')
        print(60*'-')
        print(10*' ' + f'Test Images From CVC-ClinicDB Dataset are => {len(test_images_cvc_clinicdb)}')
        print(10*' ' + f'Test Images With GT-Object From CVC-ClinicDB Dataset are => {len(test_images_cvc_clinicdb_gt_object)}')
        print(60*'-')
        print(10*' ' + f'Test Images From CVC-ColonDB Dataset are => {len(test_images_cvc_colondb)}')
        print(10*' ' + f'Test Images With GT-Object From CVC-ColonDB Dataset are => {len(test_images_cvc_colondb_gt_object)}')
        print(60*'-')
        print(10*' ' + f'Test Images From ETIS Dataset are => {len(test_images_etis)}')
        print(10*' ' + f'Test Images With GT-Object From ETIS Dataset are => {len(test_images_etis_gt_object)}')
        print(60*'-')
        print(10*' ' + f'Test Images From KVASIR Dataset are => {len(test_images_kvasir)}')
        print(10*' ' + f'Test Images With GT-Object From KVASIR Dataset are => {len(test_images_kvasir_gt_object)}')
        print(60*'-')
    
    dictionary_of_polyp_evaluation = {}
    dictionary_of_polyp_evaluation['cvc_300_evaluate_test_images'] = test_images_cvc_300
    dictionary_of_polyp_evaluation['cvc_300_evaluate_gt_object_images'] = test_images_cvc_300_gt_object
    dictionary_of_polyp_evaluation['cvc_clinicdb_evaluate_test_images'] = test_images_cvc_clinicdb
    dictionary_of_polyp_evaluation['cvc_clinicdb_evaluate_gt_object_images'] = test_images_cvc_clinicdb_gt_object
    dictionary_of_polyp_evaluation['cvc_colondb_evaluate_test_images'] = test_images_cvc_colondb
    dictionary_of_polyp_evaluation['cvc_colondb_evaluate_gt_object_images'] = test_images_cvc_colondb_gt_object
    dictionary_of_polyp_evaluation['etis_evaluate_test_images'] = test_images_etis
    dictionary_of_polyp_evaluation['etis_evaluate_gt_object_images'] = test_images_etis_gt_object
    dictionary_of_polyp_evaluation['kvasir_evaluate_test_images'] = test_images_kvasir
    dictionary_of_polyp_evaluation['kvasir_evaluate_gt_object_images'] = test_images_kvasir_gt_object

    return train_images, train_images_gt_object, dictionary_of_polyp_evaluation