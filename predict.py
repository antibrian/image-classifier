from ICM import ICM
import argparse

parser = argparse.ArgumentParser(description='Image Classifier Predictor')

parser.add_argument('image_path', help="path to test image")
parser.add_argument('--checkpoint', dest='checkpoint_path', default='checkpoint.pth', help="path to checkpoint")
parser.add_argument('--category_names', dest='json_path', help="path to json containing category names")
parser.add_argument('--top_k', dest="top_k", type=int, default=5, help="number of predictions to display")
parser.add_argument('--gpu', dest="gpu", action="store_true", default=False, help="use GPU for computation, if available")

locals().update(vars(parser.parse_args()))

image_classifier = ICM(gpu=gpu)
image_classifier.load_checkpoint(checkpoint_path)
if json_path is not None:
    image_classifier.set_category_names(json_path)
    
probs, classes, labels = image_classifier.predict(image_path)

for line in zip(probs, classes, labels):
    print('{:2.2%} {} {:>30}'.format(*line))
