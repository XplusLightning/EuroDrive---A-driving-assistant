import torch
import config
from config import args_setting
from model import generate_model
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from mss import mss
from LaneControl import LaneControler
from GeneralCVFunctions import GeneralFunctions 


def output_result(model, data, device):
    model.eval()
    feature_dic=[]
    with torch.no_grad():
        tensor = data.to(device)
        output,feature = model(tensor[None, ...])
        feature_dic.append(feature)
        pred = output.max(1, keepdim=True)[1]
        img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1,-1,3).numpy()*255
        img = Image.fromarray(img.astype(np.uint8))
        return np.array(img)



def get_parameters(model, layer_name):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.UpsamplingBilinear2d
    )
    for name, module in model.named_children():
        if name in layer_name:
            for layer in module.children():
                if isinstance(layer, modules_skipped):
                    continue
                else:
                    for parma in layer.parameters():
                        yield parma


def screenshot():
        bounding_box = {'top': 450, 'left': 650, 'width': 600, 'height': 300}
        sct = mss()
        sct_img = sct.grab(bounding_box)
        frame = Image.frombytes(
                'RGB', 
                (sct_img.width, sct_img.height), 
                sct_img.rgb, 
            )
        return frame.resize((256, 128))


if __name__ == '__main__':
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    old_left_lane, old_right_lane = [0, 0, 0, 0], [0, 0, 0, 0]
    old_angle = 00
    general = GeneralFunctions()
    controler = LaneControler()
    live = True
    steering = False
    
    OpenCV_Method = False
    Mean_Method = False
    Barrier_Method = True
    
    op_tranforms = transforms.Compose([transforms.ToTensor()])
    model = generate_model(args)
    class_weight = torch.Tensor(config.class_weight)
    pretrained_dict = torch.load(config.pretrained_path)
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)


    while True:
        image = screenshot()
        
        if args.model == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
            data = op_tranforms(image).unsqueeze(0)
        output = output_result(model, data, device)
        lanes = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        
        
        # OpenCV Method
        if OpenCV_Method:
            left_lane, right_lane = general.approximate_lane_lines(lanes)
            if left_lane == [0, 0, 0, 0]:
                left_lane_image = general.draw_lines(lanes, np.array([old_left_lane]))
            else:
                left_lane_image = general.draw_lines(lanes, np.array([left_lane]))
                old_left_lane = left_lane.copy()
                
            if right_lane == [0, 0 ,0 ,0]:
                right_lane_image = general.draw_lines(lanes, np.array([old_right_lane]))
            else:
                right_lane_image = general.draw_lines(lanes, np.array([right_lane]))
                old_right_lane = right_lane.copy()
            
            lane_image = cv2.addWeighted(left_lane_image, 0.8, right_lane_image, 1, 1)
            cv_overlayed_lanes = cv2.addWeighted(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA), 0.8, lane_image, 1, 1)
            cv_angle = controler.lane_control(left_lane, right_lane, steering)
            cv_lane_gui = general.lane_gui(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA), cv_angle)
            cv2.imshow('cv_lane_gui', cv_lane_gui)
            cv2.imshow('cv_lane_image', cv_overlayed_lanes)
        
        # Mean Method
        if Mean_Method:
            mean_lane, slope, y_intercept = general.mean_line(lanes)
            mean_angle = controler.lane_control_mean(slope, steering)
            mean_overlayed_lanes = cv2.addWeighted(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA), 0.8, cv2.cvtColor(mean_lane, cv2.COLOR_GRAY2BGRA), 1, 1)
            mean_lane_gui = general.lane_gui(mean_overlayed_lanes, mean_angle)
            cv2.imshow('mean', mean_lane)
            cv2.imshow('mean lane', mean_lane_gui)
        
        # Barrier Method
        if Barrier_Method:
            barriers = general.barrier_check(lanes)
            middle_of_barriers, new_angle, intersect, filtered_barriers = general.middle_of_barriers(barriers)
            old_angle, steering_direction = controler.lane_control_barriers(new_angle, old_angle, intersect, steering)
            overlayed_lanes = cv2.addWeighted(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA), 0.8, cv2.cvtColor(middle_of_barriers, cv2.COLOR_GRAY2BGRA), 1, 1)
            lane_gui = general.lane_gui(overlayed_lanes, old_angle, intersect, steering_direction)
            cv2.imshow('barrier', barriers)
            cv2.imshow('middle', middle_of_barriers)
            cv2.imshow('lane_image', lane_gui)
            cv2.imshow('barrier line', filtered_barriers)
        

        cv2.imshow('original', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        cv2.imshow('lanes', lanes)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    