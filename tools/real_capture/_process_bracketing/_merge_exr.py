import rawpy

def ReadImage(fileName):
    if os.path.splitext(fileName)[1].lower() == '.dng':
        raw = rawpy.imread(fileName)
        image = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=8)
        return image.astype(np.uint8)[:, :, ::-1]
    return cv.resize(cv.imread(fileName), (3600, 1800))
        

def ConvertHDR(
        fileNames: list,
        exposures: list,
        exposureTarget: float
        ):
    images = [ReadImage(fileNeme) for fileNeme in fileNames]
    # for (exposure, image) in zip(exposures, images):
    #     cv.imshow('orig_%06f' % exposure, image)
    exposures = np.array(exposures, dtype=np.float32)
    if False:
        calibrateDebevec = cv.createCalibrateDebevec()
        calibrateDebevec = calibrateDebevec.process(images, times=exposures)
        plt.plot(calibrateDebevec[:, 0, 0], c='b')
        plt.plot(calibrateDebevec[:, 0, 1], c='g')
        plt.plot(calibrateDebevec[:, 0, 2], c='r')
        plt.show()
    images = np.array(images, dtype=np.float32) / np.array(255, dtype=np.float32)
    np.clip(images, 0.0, 1.0, out=images)
    # print(images.shape)
    weights = np.minimum(images, 1.0 - images)
    np.clip(weights, 0.0, 1.0, out=weights)
    weights[0][images[0]<0.5] = 1.0
    weights[-1][images[-1]>0.5] = 1.0
    exposureTarget = np.array(exposureTarget, dtype=np.float32)
    images = images * (exposureTarget / exposures)[:, np.newaxis, np.newaxis, np.newaxis]
    # print(weights.dtype)
    imageHDR = np.sum(images * weights, axis=0) / np.sum(weights, axis=0)
    imageHDR = ndimage.median_filter(imageHDR, size=3)[::3, ::3]
    # print(imageHDR.shape)
    # for exposure in exposures:
    #     cv.imshow('hdr_%06f' % exposure, imageHDR * exposure / exposureTarget)
    # cv.waitKey(0)
    return imageHDR