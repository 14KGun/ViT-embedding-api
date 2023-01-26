import numpy as np
from utils import loadImagesFromPath, loadImagesFromUrl, embedding, maxpooling, similarity, similarityImage

if __name__ == "__main__":
    imageUrls = [
        'http://images.cocodataset.org/val2017/000000039769.jpg', # 고양이 사진 1
        'https://blog.kakaocdn.net/dn/tEMUl/btrDc6957nj/NwJoDw0EOapJNDSNRNZK8K/img.jpg', # 고양이 사진 2
        'http://www.ikunkang.com/news/photo/202009/32320_21987_1540.jpg', # 강아지 사진
        'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F9962DE405A8280FD23', # 풍경 사진
        'https://cdn.pixabay.com/photo/2020/06/02/06/52/cat-5249722__340.jpg', # 고양이 사진 3
    ]
    imagePaths = [
        "test-image/mandoo1.jpg",
        "test-image/mandoo2.jpg",
        "test-image/mandoo3.jpg",
        "test-image/mandoo4.jpg",
        "test-image/mandoo5.jpg",
    ]
    imageVecs = embedding(loadImagesFromPath(imagePaths))
    imageVecs = [np.asarray(vec) for vec in imageVecs]
    imageVec2s = embedding(loadImagesFromUrl(imageUrls))
    imageVec2s = [np.asarray(vec) for vec in imageVec2s]
    images = loadImagesFromPath(imagePaths)
    image2s = loadImagesFromUrl(imageUrls)

    # print("만두 대표 이미지 유사도")
    # print(similarity(globalVec, imageVecs[0]))
    # print(similarity(globalVec, imageVecs[1]))
    # print(similarity(globalVec, imageVecs[2]))
    # print(similarity(globalVec, imageVecs[3]))
    # print(similarity(globalVec, imageVecs[4]))

    print("만두 유사도")
    globalVec = maxpooling(imageVecs)
    print(similarity(imageVecs[0], imageVecs[1]), similarityImage(images[0], images[1])) # 0.003955771506401456 45
    print(similarity(imageVecs[1], imageVecs[2]), similarityImage(images[1], images[2])) # 0.0040384127802758464 20
    print(similarity(imageVecs[2], imageVecs[3]), similarityImage(images[2], images[3])) # 0.0038210871208565833 15
    print(similarity(imageVecs[3], imageVecs[4]), similarityImage(images[3], images[4])) # 0.010270610044543322 99
    
    print("서로 다른 이미지 유사도")
    print(similarity(imageVec2s[0], imageVec2s[1]), similarityImage(image2s[0], image2s[1])) # 0.003185808273233082 3
    print(similarity(imageVec2s[1], imageVec2s[2]), similarityImage(image2s[1], image2s[2])) # 0.0027021158853073155 20
    print(similarity(imageVec2s[2], imageVec2s[3]), similarityImage(image2s[2], image2s[3])) # 0.0027682038803065025 10
    print(similarity(imageVec2s[3], imageVec2s[4]), similarityImage(image2s[3], image2s[4])) # 0.0028074110757701724 13
