from ICM import ICM
import argparse

parser = argparse.ArgumentParser(description='Image Classifier Trainer')

parser.add_argument('data_dir', help="directory containing testing, validation, and training images")
parser.add_argument('--checkpoint', dest="checkpoint", help="load pre-existing checkpoint for further training")
parser.add_argument('--arch', dest="network", default="resnet50", help="pre-trained feature set model name (default: resnet50)")
parser.add_argument('--save_dir', dest="save_dir", help="directory in which to save checkpoint")
parser.add_argument('--learning_rate', dest="learning_rate", type=float, default=0.001, help="optimizer learning rate (default: .001)")
parser.add_argument('--hidden_units', dest="hidden_units", type=int, default=2, help="number of hidden units (default: 2)")
parser.add_argument('--epochs', dest="epochs", type=int, default=0, help="number of training passes. 0 for no training")
parser.add_argument('--gpu', dest="gpu", action="store_true", default=False, help="use GPU for computation, if available")
parser.add_argument('--test', dest="test", action="store_true", default=False, help="run tests against testing images")
parser.add_argument('--no-save', dest="save", action="store_false", default=True, help='do not save a checkpoint')

locals().update(vars(parser.parse_args()))

image_classifier = ICM(network=network, learning_rate=learning_rate, hidden_units=hidden_units, gpu=gpu)

if checkpoint is not None:
    image_classifier.load_checkpoint(checkpoint)
else:
    image_classifier.build_network()

image_classifier.load_data(data_dir)

if epochs > 0:
    image_classifier.train(epochs)

if test: image_classifier.test()
if save: image_classifier.checkpoint(save_dir)
