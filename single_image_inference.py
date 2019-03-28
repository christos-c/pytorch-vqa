import json

import skimage
import cv2
import numpy
import torch
from PIL import Image
from torch.autograd import Variable

import config
import model
import utils
from resnet import resnet as caffe_resnet


class ImgNet(torch.nn.Module):
    def __init__(self):
        super(ImgNet, self).__init__()
        self.model = caffe_resnet.resnet152(pretrained=True)

        # noinspection PyUnusedLocal
        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


def prepare_data(question, img):
    question = question.lower()[:-1].split(' ')
    img_var = Variable(img.cuda(async=True))
    img_feats = img_net(img_var.unsqueeze(0))
    question_vec = torch.zeros(max_question_length).long()
    for i, token in enumerate(question):
        index = token_to_index.get(token, 0)
        question_vec[i] = index
    question_var = Variable(question_vec.cuda(async=True))
    question_len_tensor = torch.from_numpy(numpy.array([len(question)]))
    question_len_var = Variable(question_len_tensor.cuda(async=True))
    return img_feats, question_var.unsqueeze(0), question_len_var


def run(question, img):
    img_emb, q_emb, q_len = prepare_data(question, img)

    out, att = net(img_emb, q_emb, q_len)
    # attention shape 1, 2, 14, 14 -- [0,0] will pull first facet, [0,1] will pull the second
    att_matrix = att[0, 0]
    # noinspection PyUnresolvedReferences
    att_upsampled = skimage.transform.pyramid_expand(att_matrix.data.cpu().numpy(), upscale=32)
    att_greyscale = numpy.array(255*(att_upsampled - numpy.min(att_upsampled))/numpy.ptp(att_upsampled), numpy.uint8)
    _, pred_index = out.max(dim=1, keepdim=True)
    return index_to_answer[pred_index.data[0][0].item()], att_matrix, att_greyscale


def apply_attention_to_image(cropped_img, activation_map):
    # cropped_img is a 3x448x448 tensor (CxHxW) -- need to transpose to HxWxC
    cropped_img_trans = cropped_img.permute(1, 2, 0).numpy()
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    img_with_heatmap = numpy.float32(activation_heatmap) + numpy.float32(cropped_img_trans)
    img_with_heatmap = img_with_heatmap / numpy.max(img_with_heatmap)
    return numpy.uint8(255 * img_with_heatmap)


def single_image_predict(image_path, question):
    pil_image = Image.open(image_path).convert('RGB')
    cropped_image = transform(pil_image)
    answer, attention, attention_scaled = run(question, cropped_image)
    return cropped_image, answer, attention, attention_scaled


def test():
    # Setup the image for inference
    image_path = '/home/chrchrs/vqa-data/train2014/COCO_train2014_000000144646.jpg'
    cropped_image, answer, raw_att, attention = single_image_predict(image_path, 'How many bikes are there?')
    print(answer)
    cv2.imwrite(image_path + '_hm.jpg', apply_attention_to_image(cropped_image, attention))

    cropped_image, answer, raw_att, attention = single_image_predict(image_path, 'What color are the jackets?')
    print(answer)
    cv2.imwrite(image_path + '_hm2.jpg', apply_attention_to_image(cropped_image, attention))


if __name__ == '__main__':
    # Initialise the ImageNet NN for the image embeddings
    img_net = ImgNet().cuda()
    img_net.eval()

    # Initialise the pre-trained VQA NN
    log = torch.load('/home/chrchrs/workspace/pytorch-vqa-remote/logs/2017-08-04_00.55.19.pth')
    tokens = len(log['vocab']['question']) + 1
    net = torch.nn.DataParallel(model.Net(tokens)).cuda()
    net.load_state_dict(log['weights'])
    net.eval()

    # Initialise the question/answer dictionaries
    with open(config.vocabulary_path, 'r') as fd:
        vocab_json = json.load(fd)
    transform = utils.get_transform(config.image_size, config.central_fraction)
    token_to_index = vocab_json['question']
    answer_to_index = vocab_json['answer']
    index_to_answer = {answer_to_index[key]: key for key in answer_to_index.keys()}
    max_question_length = 23

    home_dir = '/home/chrchrs/vqa-data/'
    image_dir = home_dir + 'val2014/'
    data_file = home_dir + 'joshua_chosenQns_22Mar.tsv'
    out_file = home_dir + 'joshua_chosenQns_22Mar_att.json'
    attentions = {}

    for line in open(data_file).readlines():
        image = line.split('\t')[0]
        question = line.split('\t')[1]
        cropped_image, answer, att14x14, attention_scaled = single_image_predict(image_dir+image, question)
        att_matrix = att14x14.data.cpu().numpy().tolist()
        attentions[image+'#'+question] = att_matrix
    json.dump(attentions, open(out_file, 'w'))
