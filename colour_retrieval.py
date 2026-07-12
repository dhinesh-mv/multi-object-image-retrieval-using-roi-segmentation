import cv2
import os

object_folder = "objects"

num = input("Enter object number: ")

query_path = f"{object_folder}/object_{num}.png"

query = cv2.imread(query_path)

if query is None:
    print("Object not found!")
    exit()

query_hist = cv2.calcHist([query], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
cv2.normalize(query_hist, query_hist)

results = []

for file in os.listdir(object_folder):

    path = os.path.join(object_folder, file)
    img = cv2.imread(path)

    if img is None:
        continue

    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    cv2.normalize(hist, hist)

    similarity = cv2.compareHist(query_hist, hist, cv2.HISTCMP_CORREL)

    results.append((file, similarity))

results = sorted(results, key=lambda x: x[1], reverse=True)

print("\nTop Matching Objects:\n")

for r in results[:5]:
    print(r[0], "Similarity:", round(r[1], 2))