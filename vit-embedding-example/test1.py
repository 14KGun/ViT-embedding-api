import numpy as np
from utils import loadImagesFromUrl, embedding, similarity

if __name__ == "__main__":
    imageUrls = [
        'http://images.cocodataset.org/val2017/000000039769.jpg', # 고양이 사진 1
        'https://blog.kakaocdn.net/dn/tEMUl/btrDc6957nj/NwJoDw0EOapJNDSNRNZK8K/img.jpg', # 고양이 사진 2
        'http://www.ikunkang.com/news/photo/202009/32320_21987_1540.jpg', # 강아지 사진
        'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F9962DE405A8280FD23', # 풍경 사진
        'https://cdn.pixabay.com/photo/2020/06/02/06/52/cat-5249722__340.jpg', # 고양이 사진 3
    ]
    imageVecs = embedding(loadImagesFromUrl(imageUrls))
    imageVecs = [np.asarray(vec) for vec in imageVecs]
    
    print("고양이 vs 고양이")
    print(similarity(imageVecs[0], imageVecs[1]))
    print(similarity(imageVecs[0], imageVecs[2]))
    print(similarity(imageVecs[0], imageVecs[4]))
    print(similarity(imageVecs[2], imageVecs[4]))
    print("고양이 vs 강아지")
    print(similarity(imageVecs[1], imageVecs[2]))
    print("동물 vs 풍경")
    print(similarity(imageVecs[0], imageVecs[3]))
    print(similarity(imageVecs[1], imageVecs[3]))
    print(similarity(imageVecs[2], imageVecs[3]))
