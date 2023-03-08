import os

def dataset_creation(show=True):
    
    # Image Paths For Train Dataset
    # Train Dataset contains 1000 Images from CAMO Dataset & 2026 Images from COD10K Dataset
    PATH_TO_IMAGES_TRAIN = './Datasets/TrainDataset/Imgs'
    PATH_TO_IMAGES_TRAIN_GT_Object = './Datasets/TrainDataset/GT'
    
    # Image Paths For Test From COD10K Dataset
    PATH_TO_IMAGES_TEST_COD10K = './Datasets/TestDataset/COD10K/Imgs'
    PATH_TO_IMAGES_TEST_COD10K_GT_Object = './Datasets/TestDataset/COD10K/GT'
    
    # Image Paths For Test From Camo Dataset
    PATH_TO_IMAGES_TEST_CAMO_DATASET = './Datasets/TestDataset/CAMO/Imgs'
    PATH_TO_IMAGES_TEST_CAMO_DATASET_GT_Object = './Datasets/TestDataset/CAMO/GT'

    # Image Paths For Test From Chameleon Dataset
    PATH_TO_IMAGES_TEST_CHAMELEON_DATASET = './Datasets/TestDataset/CHAMELEON/Imgs'
    PATH_TO_IMAGES_TEST_CHAMELEON_DATASET_GT_Object = './Datasets/TestDataset/CHAMELEON/GT'
    
    # Image Paths For Test From NC4K Dataset
    PATH_TO_IMAGES_TEST_NC4K_DATASET = './Datasets/TestDataset/NC4K/Imgs'
    PATH_TO_IMAGES_TEST_NC4K_DATASET_GT_Object = './Datasets/TestDataset/NC4K/GT'
    
    # define train variables from Train Dataset
    train_images_half_path = sorted(os.listdir(PATH_TO_IMAGES_TRAIN))
    train_images_half_path_gt_object = sorted(os.listdir(PATH_TO_IMAGES_TRAIN_GT_Object))
    
    # define test variables from COD10K Dataset
    test_images_cod10k_half_path = sorted(os.listdir(PATH_TO_IMAGES_TEST_COD10K))
    test_images_cod10k_half_path_gt_object = sorted(os.listdir(PATH_TO_IMAGES_TEST_COD10K_GT_Object))

    # define test variables from CAMO Dataset
    test_images_camo_half_path = sorted(os.listdir(PATH_TO_IMAGES_TEST_CAMO_DATASET))
    test_images_camo_half_path_gt_object = sorted(os.listdir(PATH_TO_IMAGES_TEST_CAMO_DATASET_GT_Object))

    # define test variables from CHAMELEON Dataset
    test_images_chameleon_half_path = sorted(os.listdir(PATH_TO_IMAGES_TEST_CHAMELEON_DATASET))
    test_images_chameleon_half_path_gt_object = sorted(os.listdir(PATH_TO_IMAGES_TEST_CHAMELEON_DATASET_GT_Object))

    # define test variables from NC4K Dataset
    test_images_nc4k_half_path = sorted(os.listdir(PATH_TO_IMAGES_TEST_NC4K_DATASET))
    test_images_nc4k_half_path_gt_object = sorted(os.listdir(PATH_TO_IMAGES_TEST_NC4K_DATASET_GT_Object))

    # Train Dataset
    train_images = []
    train_images_gt_object = []
    
    # Test Dataset
    # COD10K
    test_images_cod10k = []
    test_images_cod10k_gt_object = []

    # CHAMELEON 
    test_images_chameleon = []
    test_images_chameleon_gt_object  = []

    # NC4K
    test_images_nc4k = []
    test_images_nc4k_gt_object  = []
    
    # CAMO 
    test_images_camo = []
    test_images_camo_gt_object  = []

    # insert the full path to train images from Train Dataset
    for i in range(len(train_images_half_path)):
        train_images.append(PATH_TO_IMAGES_TRAIN + "/{}".format(train_images_half_path[i]))
        train_images_gt_object.append(PATH_TO_IMAGES_TRAIN_GT_Object + "/{}".format(train_images_half_path_gt_object[i]))
        
    # insert the full path to each test image from COD10K Dataset
    for i in range(len(test_images_cod10k_half_path)):
        test_images_cod10k.append(PATH_TO_IMAGES_TEST_COD10K + "/{}".format(test_images_cod10k_half_path[i]))
        test_images_cod10k_gt_object.append(PATH_TO_IMAGES_TEST_COD10K_GT_Object + "/{}".format(test_images_cod10k_half_path_gt_object[i]))
                
    # insert the full path to each test image from CHAMELEON Dataset        
    for i in range(len(test_images_chameleon_half_path)):
        test_images_chameleon.append(PATH_TO_IMAGES_TEST_CHAMELEON_DATASET + "/{}".format(test_images_chameleon_half_path[i]))
        test_images_chameleon_gt_object.append(PATH_TO_IMAGES_TEST_CHAMELEON_DATASET_GT_Object + "/{}".format(test_images_chameleon_half_path_gt_object[i]))
    
    # insert the full path to each test image from NC4K Dataset
    for i in range(len(test_images_nc4k_half_path)):
        test_images_nc4k.append(PATH_TO_IMAGES_TEST_NC4K_DATASET + "/{}".format(test_images_nc4k_half_path[i]))
        test_images_nc4k_gt_object.append(PATH_TO_IMAGES_TEST_NC4K_DATASET_GT_Object + "/{}".format(test_images_nc4k_half_path_gt_object[i]))
    
    # insert the full path to each train-test image from CAMO Dataset
    for i in range(len(test_images_camo_half_path)):
        test_images_camo.append(PATH_TO_IMAGES_TEST_CAMO_DATASET + "/{}".format(test_images_camo_half_path[i]))
        test_images_camo_gt_object.append(PATH_TO_IMAGES_TEST_CAMO_DATASET_GT_Object + "/{}".format(test_images_camo_half_path_gt_object[i]))
    
    if show:
        print('[INFO] Initializing Train Dataset composed by COD10K + CAMO')
        print(60*'-')
        print(10*' ' + f'Train Images From Train Dataset are => {len(train_images)}')
        print(10*' ' + f'Train Images With GT-Object are => {len(train_images_gt_object)}')
        print(60*'-')
        print('[INFO] Creating Test Dataset composed by COD10K + CAMO + CHAMELEON + NC4K')
        print(60*'-')
        print(10*' ' + f'Test Images From COD10K Dataset are => {len(test_images_cod10k)}')
        print(10*' ' + f'Test Images With GT-Object From COD10K Dataset are => {len(test_images_cod10k_gt_object)}')
        print(60*'-')
        print(10*' ' + f'Test Images From CHAMELEON Dataset are => {len(test_images_chameleon)}')
        print(10*' ' + f'Test Images With GT-Object From CHAMELEON Dataset are => {len(test_images_chameleon_gt_object)}')
        print(60*'-')
        print(10*' ' + f'Test Images From NC4K Dataset are => {len(test_images_nc4k)}')
        print(10*' ' + f'Test Images With GT-Object From NC4K Dataset are => {len(test_images_nc4k_gt_object)}')
        print(60*'-')
        print(10*' ' + f'Test Images From CAMO Dataset are => {len(test_images_camo)}')
        print(10*' ' + f'Test Images With GT-Object From CAMO Dataset are => {len(test_images_camo_gt_object)}')
        print(60*'-')
                
    test_images = test_images_cod10k + test_images_chameleon + test_images_nc4k + test_images_camo
    test_images_gt_object = test_images_cod10k_gt_object + test_images_chameleon_gt_object + test_images_nc4k_gt_object + test_images_camo_gt_object
    
    if show:
        print('[INFO] Test Images Have Been Created From COD10K + CHAMELEON + NC4K + CAMO Datasets')
        print(60*'-')
        print(10*' ' + f'Test Images From Mixed Dataset are => {len(test_images)}')
        print(10*' ' + f'Test Images With GT-Object From Mixed Dataset are => {len(test_images_gt_object)}')
        print(60*'-' + f'\n')
    
    dictionary_of_evaluation = {}
    dictionary_of_evaluation['chameleon_evaluate_test_images'] = test_images_chameleon
    dictionary_of_evaluation['chameleon_evaluate_gt_object_images'] = test_images_chameleon_gt_object
    dictionary_of_evaluation['camo_evaluate_test_images'] = test_images_camo
    dictionary_of_evaluation['camo_evaluate_gt_object_images'] = test_images_camo_gt_object
    dictionary_of_evaluation['cod10k_evaluate_test_images'] = test_images_cod10k
    dictionary_of_evaluation['cod10k_evaluate_gt_object_images'] = test_images_cod10k_gt_object
    dictionary_of_evaluation['nc4k_evaluate_test_images'] = test_images_nc4k
    dictionary_of_evaluation['nc4k_evaluate_gt_object_images'] = test_images_nc4k_gt_object
    
    return train_images, train_images_gt_object, test_images, test_images_gt_object, dictionary_of_evaluation