import cv2, os
from keras.models import load_model

img_size = 258
base_path = 'samples'
file_list = sorted(os.listdir(base_path))

model_name = "kat_model_features_0"
model = load_model(model_name) #load model

# testing each image in samples folder
for f in file_list:
    if '.jpg' not in f:
        continue

    img = cv2.imread(os.path.join(base_path, f))

    # resize image for testing in model
    img = cv2.resize(img, (img_size, img_size))
    face_inputs = (img.astype('float32') / 255).reshape(-1, img_size, img_size, 3)

    predicted_features = model.predict(face_inputs) # predict facial feature points

    # itterate though all facial feature points in the prediction and draw on image
    for i in range(1, 18, 2):
        center_point = (int(predicted_features[0][i]), int(predicted_features[0][i+1]))
        cv2.circle(img, center_point, radius=10, color=(255, 0, 0), thickness=5)

    #display image
    cv2.imshow('img', img)
    filename, ext = os.path.splitext(f)

    # save results to results folder
    cv2.imwrite('result/%s_result%s' % (filename, ext), img)

    if cv2.waitKey(0) == ord('q'): # go to next picture
        cv2.destroyAllWindows()
    if cv2.waitKey(1) == ord('e'): # exit
        break