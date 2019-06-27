from __future__ import absolute_import
import torch
from torchvision import datasets,transforms
from PIL import Image
import numpy as np
from scipy import misc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import os
from network import init_network
import errno

def prepare_data_lst(dataset,small_img_size,normalize=False):
    img_lst = []
    small_img_lst=[]
    label_lst = []
    for _, file in enumerate(dataset.imgs):
        path, label = file
        original_img = Image.open(path).convert('RGB')
        img = original_img.resize((128, 256))
        small_img=original_img.resize(small_img_size)
        img = transforms.ToTensor()(img)
        small_img=transforms.ToTensor()(small_img)
        if normalize==True:
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])(img)
        img_lst.append(img)
        small_img_lst.append(small_img)
        label_lst.append(label)
    img_tensor = torch.stack(img_lst, 0)
    small_img_tensor=torch.stack(small_img_lst,0)
    label=np.array(label_lst)
    return img_tensor, label,small_img_tensor

def sample(img_tensor,label,num_sampled_id,small_img):
    index=label<num_sampled_id
    sampled_label=label[index]
    sampled_img=img_tensor.numpy()[index,:,:,:]
    sampled_small_img=small_img.numpy()[index,:,:,:]
    assert (sampled_img.shape[0]==sampled_label.shape[0]==sampled_small_img.shape[0])
    return sampled_img,sampled_label,sampled_small_img

def plot_person_img(sampled_img,n_img_per_row,save_dir):

    n_samples,channel,height,width=sampled_img.shape
    large_img=np.zeros((channel,n_img_per_row*height,n_img_per_row*width))
    for i in range(n_img_per_row):
        x=height*i
        for j in range(n_img_per_row):
            y=width*j
            large_img[:,x:x+height,y:y+width]=sampled_img[i*n_img_per_row+j,:,:,:]
    large_img=np.transpose(large_img,(1,2,0))
    misc.imsave(os.path.join(save_dir,'person_selected.png'),large_img)

def load_checkpoint(path,model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['shared_state_dict'])
    return model

def extract_feature(input,model):
    _, output = model(input)
    output = output[4]
    output = output.view(output.size(0), -1)
    return output

def plot_tsne_embedding(tsne_embedding,label,save_dir):
    min_value=np.min(tsne_embedding,0)
    max_value=np.max(tsne_embedding,0)
    tsne_embedding=(tsne_embedding-min_value)/(max_value-min_value)
    num_id=len(set(list(label)))
    plt.figure()
    for i in range(tsne_embedding.shape[0]):
        plt.text(tsne_embedding[i,0],tsne_embedding[i,1],str(label[i]),
                 color=plt.cm.Set1(label[i]/float(num_id)),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    plt.savefig(os.path.join(save_dir,'tsne_label.png'),dpi=120)

def plot_tsne_true_img(small_img,tsne_embedding,save_dir,scale1,scale2):
    n_samples, channel, height, width = small_img.shape
    large_img = np.zeros((channel, scale1 * height, scale1 * width))
    min_value = np.min(tsne_embedding, 0)
    max_value = np.max(tsne_embedding, 0)
    tsne_embedding = (tsne_embedding - min_value) / (max_value - min_value)
    for i in range(tsne_embedding.shape[0]):
        x=int(tsne_embedding[i, 0]*width*scale2)
        y=int(tsne_embedding[i, 1]*height*scale2)
        large_img[:,x:x+height,y:y+width]=small_img[i,:,:,:]
    large_img = np.transpose(large_img, (1, 2, 0))
    misc.imsave(os.path.join(save_dir,'tsne_true_image.png'), large_img)

def main(args):
    # check the save dir
    try:
        os.makedirs(args.save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # initialize dataset
    dataset = datasets.ImageFolder(args.dataset_dir)

    # prepare data list
    img_tensor, label, small_img = prepare_data_lst(dataset,args.small_image_size)

    # sample
    sampled_img, sampled_label, sampled_small_img = sample(img_tensor, label, args.num_sampled_id, small_img)

    # plot sampled images
    plot_person_img(sampled_img, args.num_images_per_row,args.save_dir)

    # select model
    if args.model is not 'private':
        model=init_network(args.model,751)
    else:
        model=init_network(args.model)

    # load checkpoint
    model = load_checkpoint(args.checkpoint_dir, model)

    # prepare inputs of the network
    transformed_img_tensor = torch.Tensor(sampled_img)

    # extract feature
    print("Extracting feature")
    feature = extract_feature(transformed_img_tensor, model)

    # t-SNE embedding
    print("Computing t-SNE embedding")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_embedding = tsne.fit_transform(feature.detach().numpy())

    # plot t-SNE embedding with labels
    plot_tsne_embedding(tsne_embedding, sampled_label,args.save_dir)

    #plot t-SNE embedding with true images
    plot_tsne_true_img(sampled_small_img, tsne_embedding, args.save_dir,args.large_image_scale,args.small_image_scale)

def opt():
    parser = argparse.ArgumentParser(description="t-SNE in reid")
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--save-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'images'))

    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/home/xiaoxi.xjl/my_code/UDA/gradient-reversal/source_models/source.pt')
    parser.add_argument('--dataset-dir', type=str,
                        default="/home/xiaoxi.xjl/re_id_dataset/DukeMTMC-reID/pytorch/query")
    parser.add_argument('--num-sampled-id', type=int, default=50)
    parser.add_argument('--num-images-per-row', type=int, default=10)
    parser.add_argument('--small-image-size', type=tuple, default=(32,64))
    parser.add_argument('--small-image-scale', type=int, default=50)
    parser.add_argument('--large-image-scale', type=int, default=200)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=opt()
    main(args)


