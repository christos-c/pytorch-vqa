import json
import os
import re

import skimage
import cv2
import numpy
import torch
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

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
    return index_to_answer[pred_index.data[0][0]], att_greyscale


def apply_attention_to_image(cropped_img, activation_map):
    # cropped_img is a 3x448x448 tensor (CxHxW) -- need to transpose to HxWxC
    cropped_img_trans = cropped_img.permute(1, 2, 0).numpy()
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    img_with_heatmap = numpy.float32(activation_heatmap) + numpy.float32(cropped_img_trans)
    img_with_heatmap = img_with_heatmap / numpy.max(img_with_heatmap)
    return numpy.uint8(255 * img_with_heatmap)


# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def process_punctuation(s):
    # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
    # this version should be faster since we use re instead of repeated operations on str's
    if _punctuation.search(s) is None:
        return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '')
    s = _punctuation.sub(' ', s)
    s = _period_strip.sub('', s)
    return s.strip()


if __name__ == '__main__':
    project_path = '/home/chrchrs/workspace/pytorch-vqa-remote'

    # Initialise the ImageNet NN for the image embeddings
    img_net = ImgNet().cuda()
    img_net.eval()

    # Initialise the pre-trained VQA NN
    log = torch.load('%s/logs/2017-08-04_00.55.19.pth' % project_path)
    tokens = len(log['vocab']['question']) + 1
    net = torch.nn.DataParallel(model.Net(tokens)).cuda()
    net.load_state_dict(log['weights'])
    net.eval()

    # Initialise the question/answer dictionaries
    with open('%s/%s' % (project_path, config.vocabulary_path), 'r') as fd:
        vocab_json = json.load(fd)
    transform = utils.get_transform(config.image_size, config.central_fraction)
    token_to_index = vocab_json['question']
    answer_to_index = vocab_json['answer']
    index_to_answer = {answer_to_index[key]: key for key in answer_to_index.keys()}
    max_question_length = 23

    # Load validation questions & answers
    data_path = '/home/chrchrs/vqa-data'
    questions_path = '%s/MultipleChoice_mscoco_val2014_questions.json' % data_path
    answers_path = '%s/mscoco_val2014_annotations.json' % data_path
    with open(questions_path, 'r') as fd:
        questions_json = json.load(fd)
    id_to_question = {q['question_id']: q['question'] for q in questions_json['questions']}
    with open(answers_path, 'r') as fd:
        answers_json = json.load(fd)
    id_to_answers = {ans_dict['question_id']: [process_punctuation(a['answer']) for a in ans_dict['answers']]
                     for ans_dict in answers_json['annotations']}

    # Load validation images dictionary
    val_images_path = '/home/chrchrs/vqa-data/val2014'
    id_to_filename = {}
    for filename in os.listdir(val_images_path):
        if not filename.endswith('.jpg'):
            continue
        # Ignore the heatmap images
        if filename.endswith('_hm.jpg'):
            continue
        id_and_extension = filename.split('_')[-1]
        id = int(id_and_extension.split('.')[0])
        id_to_filename[id] = filename

    # Run the inference for all questions in the dataset and write out the heatmaps
    with open('%s/MultipleChoice_mscoco_val2014_output.tsv' % data_path, 'w') as out:
        for q_id in tqdm(id_to_question):
            question = id_to_question[q_id]
            # Skip questions we don't have gold answers for
            if q_id not in id_to_answers:
                continue
            # Skip questions we don't have images for
            if q_id not in id_to_filename:
                continue

            image_path = '%s/%s' % (val_images_path, id_to_filename[q_id])
            # Setup the image for inference
            pil_image = Image.open(image_path).convert('RGB')
            cropped_image = transform(pil_image)

            # Run the inference
            answer, attention = run(question, cropped_image)
            gold_answers = id_to_answers[q_id]
            out.write('%s\t\%s\t%s\t%s\n' % (id_to_filename[q_id], question, answer, '\t'.join(gold_answers)))

            cv2.imwrite(image_path + '_hm.jpg', apply_attention_to_image(cropped_image, attention))
