import frameSelect.hyperIQA.models as models
import torch 
import torchvision
from PIL import Image
import numpy as np
import statistics

class hyperIQA:
    def __init__(self):
        self.methodName = 'HyperIQA'
        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_hyper.train(False)
        self.model_hyper.load_state_dict((torch.load('pipeline/frameSelect/hyperIQA/pretrained/koniq_pretrained.pkl',weights_only=True)))
        self.transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        self.name()
    def pil_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def predict(self,im_path, mute = False):
        pred_scores = []
        for i in range(10):
            img = self.pil_loader(im_path)
            img = self.transforms(img)
            img_cuda = img.cuda()
            img = img_cuda.clone().detach().unsqueeze(0)
            paras = self.model_hyper(img)  # 'paras' contains the network weights conveyed to target network

            # Building target network
            model_target = models.TargetNet(paras).cuda()
            for param in model_target.parameters():
                param.requires_grad = False

                # Quality prediction
            pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
            pred_scores.append(float(pred.item()))
        score = np.mean(pred_scores)
        # quality score ranges from 0-100, a higher score indicates a better quality
        if not mute:
            print('Predicted quality score: %.2f' % score)
        return score
    
    def predict_list(self, im_path_list, mute = False):
        score_list = []
        for item in im_path_list:
            score = self.predict(item, mute)
            score_list.append(score)
        self.__printPredictionSummary__(score_list)

        return score_list
    
    def __printPredictionSummary__(self,score_list):
        print(f'{len(score_list)} prediction has been made.')
        print(f'Mean Score: {sum(score_list)/len(score_list):.2f}')
        print(f'Max Score: {max(score_list):.2f}')
        print(f'Min Score: {min(score_list):.2f}')
        print(f'Standard Deviation of Scores: {statistics.stdev(score_list):.2f}')
        

    def name(self):
        print('HyperIQA has been chosen to be the method for frame selection')

    def setup(self):

        print('HyperIQA finished setting up')