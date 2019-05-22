import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

idxtow=None
graph=None
outputs=None
tensors=None
def open_files():
    global idxtow,graph,outputs,tensors
    with open('E:/Work/Projects/Image Captioning/Trained Models/merged_frozen_graph_FILE.pb', 'rb') as f:
        fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    tf.import_graph_def(graph_def, input_map=None, return_elements=None, name='', op_dict=None, producer_op_list=None)
    graph = tf.get_default_graph()
    tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
    wtoidx = np.load("E:/Work/Projects/Image Captioning/Trained Models/wordmap.npy",allow_pickle=True).tolist()
    idxtow = dict(zip(wtoidx.values(), wtoidx.keys()))

    with open('E:/Work/Projects/Image Captioning/Trained Models/DecoderOutputs.txt', 'r') as fr:
        outputs = fr.read()
        outputs = outputs.split('\n')[:-1]


def IDs_to_Words(ID_batch):
    return [idxtow[word] for IDs in ID_batch for word in IDs]



# def load_image(path, caption):
#     plt.imshow(Image.open(path))
#     plt.axis("off")
#     plt.title(caption, fontsize='10', loc='left')
#     arr = path.split("/")
#     plt.savefig("../results/gen_" + arr[-1].split('.')[0] + ".png")
#     plt.show()


in1 = None
out1 = None
in2 = None
sentence = None


def get_tensors():
    global in1, out1, in2, sentence
    in1 = graph.get_tensor_by_name("encoder/InputFile:0")
    out1 = graph.get_tensor_by_name("encoder/Preprocessed_JPG:0")
    in2 = graph.get_tensor_by_name("encoder/import/InputImage:0")
    sentence = []
    for i, outs in enumerate(outputs):
        sentence.append(graph.get_tensor_by_name("decoder/" + outs + ":0"))


def init_caption_generator():
    sess = tf.Session()
    get_tensors()
    return sess


def preprocess_image(sess, image_path):
    global in1, out1
    if image_path.split(".")[-1] == "png":
        out1 = graph.get_tensor_by_name("encoder/Preprocessed_PNG:0")
    feed_dict = {in1: image_path}
    prepro_image = sess.run(out1, feed_dict=feed_dict)
    return prepro_image
captionText= []

def generate_caption(sess, image_path):
    global in2, out1, sentence
    global captionText
    prepro_image = preprocess_image(sess, image_path)

    feed_dict = {in2: prepro_image}
    prob = sess.run(sentence, feed_dict=feed_dict)
    # set default back to JPG
    out1 = graph.get_tensor_by_name("encoder/Preprocessed_JPG:0")
    caption = " ".join(IDs_to_Words(prob)).split("</S>")[0]
    captionText.append(caption)
    print(caption)
    print("\n")
    #load_image(image_path, caption)

sess=None

def generate_caption_init():
    global sess
    open_files()
    sess = init_caption_generator()

def generate_caption_live():
    global sess
    generate_caption(sess,"Images/image.png")

import cv2
def generate_caption_bulk():
    path = "Images/"
    files = sorted(os.listdir(path))
    files = [path + f for f in files]
    for f in files:
        if os.path.splitext(f)[1] in [".png", ".jpg", ".jpeg"]:
            generate_caption(sess, f)
    caption_write()
def caption_write():
    global captionText
    font = cv2.FONT_HERSHEY_SIMPLEX
    path = "Images/"
    path2= "Captions/"
    files = sorted(os.listdir(path))
    files = [path + f for f in files]
    image_number = 1
    for f in files:
        image = cv2.imread(f)
        height, width, channel = image.shape
        cv2.putText(image,captionText[image_number-1], (12, height - 12), font, 1, (0, 0, 0), 2)
        cv2.imwrite(path2 + str(image_number) + "_captioned.jpg", image)
        image_number = image_number + 1