import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

class RotationInvariantLBP:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image):
        lbp_image = np.zeros_like(image, dtype=np.uint8)
        height, width = image.shape

        for y in range(self.radius, height - self.radius):
            for x in range(self.radius, width - self.radius):
                center_pixel = image[y, x]
                binary_pattern = 0

                for i in range(self.num_points):
                    angle = (2 * np.pi * i) / self.num_points
                    x_offset = int(self.radius * np.cos(angle))
                    y_offset = int(self.radius * np.sin(angle))

                    neighbor_pixel = image[y + y_offset, x + x_offset]

                    if neighbor_pixel >= center_pixel:
                        binary_pattern |= (1 << i)

                # 회전 불변 LBP 패턴 계산
                min_pattern = binary_pattern
                for i in range(1, self.num_points):
                    rotated_pattern = (binary_pattern >> i) | (binary_pattern << (self.num_points - i))
                    if rotated_pattern < min_pattern:
                        min_pattern = rotated_pattern
                
                 # 2^p를 할당
                    lbpp_r_value = min_pattern * (2 ** i)

                lbp_image[y, x] = lbpp_r_value

        return lbp_image

def extract_features(image_paths, labels):
    rlbp = RotationInvariantLBP(num_points=8, radius=1)
    features = []
    lbls = []
    
    for path, label in zip(image_paths, labels):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        rlbp_image = rlbp.describe(image)
        features.append(rlbp_image.flatten())
        lbls.append(label)
    
    return np.array(features), np.array(lbls)

# 1000개의 사진이 있는 디렉토리 경로
data_dir = "photos_data"

# 파일 경로와 해당 레이블 수집
image_paths = []
labels = []

for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image_paths.append(image_path)
            labels.append(class_name)

# RLBP 특징 추출 및 레이블 인코딩
features, labels = extract_features(image_paths, labels)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 데이터를 학습 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# 분류기 초기화 (SVM)
classifier = SVC(kernel='linear')

# 분류기 학습
classifier.fit(X_train, y_train)

# 테스트 세트에 대한 레이블 예측
y_pred = classifier.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("정확도:", accuracy)

# 선택적: 유사 이미지 찾기
# 특징 간의 코사인 유사도 행렬 계산
cos_sim_matrix = cosine_similarity(features)

# 쿼리 이미지 인덱스 선택
query_image_index = 0
query_features = features[query_image_index]
cos_sim_scores = cos_sim_matrix[query_image_index]

# 유사도 점수를 정렬하고 인덱스를 가져옴
similar_image_indices = np.argsort(cos_sim_scores)[::-1][:5]  # 상위 5개 유사 이미지
print("쿼리 이미지와 유사한 이미지:", similar_image_indices)