import torch
import os
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from helper import extract_patient_ids, ChexpertSmall

@torch.enable_grad()
def grad_cam(model, x, hooks, cls_idx=None):
    """ cf CheXpert: Test Results / Visualization; visualize final conv layer, using grads of final linear layer as weights,
    and performing a weighted sum of the final feature maps using those weights.
    cf Grad-CAM https://arxiv.org/pdf/1610.02391.pdf """

    model.eval()
    model.zero_grad()

    # register backward hooks
    conv_features, linear_grad = [], []
    forward_handle = hooks['forward'].register_forward_hook(lambda module, in_tensor, out_tensor: conv_features.append(out_tensor))
    backward_handle = hooks['backward'].register_backward_hook(lambda module, grad_input, grad_output: linear_grad.append(grad_input))

    # run model forward and create a one hot output for the given cls_idx or max class
    outputs = model(x)
    if not cls_idx: cls_idx = outputs.argmax(1)
    one_hot = F.one_hot(cls_idx, outputs.shape[1]).float().requires_grad_(True)

    # run model backward
    one_hot.mul(outputs).sum().backward()

    # compute weights; cf. Grad-CAM eq 1 -- gradients flowing back are global-avg-pooled to obtain the neuron importance weights
    weights = linear_grad[0][2].mean(1).view(1, -1, 1, 1)
    # compute weighted combination of forward activation maps; cf Grad-CAM eq 2; linear combination over channels
    cam = F.relu(torch.sum(weights * conv_features[0], dim=1, keepdim=True))

    # normalize each image in the minibatch to [0,1] and upscale to input image size
    cam = cam.clone()  # avoid modifying tensor in-place
    def norm_ip(t, min, max):
        tt = t.clamp(min=min, max=max)
        ttt = tt.add(-min).div(max - min + 1e-5)
        t = ttt

    for t in cam:  # loop over mini-batch dim
        norm_ip(t, float(t.min()), float(t.max()))

    cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=True)

    # cleanup
    forward_handle.remove()
    backward_handle.remove()
    model.zero_grad()

    return cam

def visualize(model, dataloader, grad_cam_hooks, device, outdir):
    attr_names = dataloader.dataset.attr_names

    # 1. run through model to compute logits and grad-cam
    # imgs, labels, scores, masks, idxs = [], [], [], [], []
    imgs, labels, scores, masks = [], [], [], []

    for ibatch, batch in enumerate(dataloader):
        length = len(batch)
        if length == 2:
            X, y = batch
            # X, y = X.to(device), y.to(device)
        elif length == 3:
            X, y, g = batch
            # X, y, g = X.to(device), y.to(device), g.to(device)
        else:
            X, y, g, ID, sp = batch
            # X, y, g = X.to(device), y.to(device), g.to(device)
        imgs += [X]
        X = X.to(device)
        labels += [y]
        masks += [grad_cam(model, X, grad_cam_hooks).cpu()]
        # scores += [model(X).cpu()]
    imgs, labels, masks = torch.cat(imgs), torch.cat(labels), torch.cat(masks)

    # for x, target, idx in dataloader:
    #     imgs += [x]
    #     labels += [target]
    #     idxs += idx.tolist()
    #     x = x.to(device)
    #     scores += [model(x).cpu()]
    #     masks  += [grad_cam(model, x, grad_cam_hooks).cpu()]
    # imgs, labels, scores, masks = torch.cat(imgs), torch.cat(labels), torch.cat(scores), torch.cat(masks)

    # 2. renormalize images and convert everything to numpy for matplotlib
    imgs.mul_(0.0349).add_(0.5330)
    imgs = imgs.permute(0,2,3,1).data.numpy()
    labels = labels.data.numpy()
    # patient_ids = extract_patient_ids(dataloader.dataset, idxs)
    masks = masks.permute(0,2,3,1).data.numpy()
    # probs = scores.sigmoid().data.numpy()

    # 3. make column grid of [model probs table, original image, grad-cam image] for each attr + other categories
    # for attr, vis_idxs in zip(dataloader.dataset.vis_attrs, dataloader.dataset.vis_idxs):
    #     fig, axs = plt.subplots(3, 3, figsize=(4 * imgs.shape[1]/100, 3.3 * imgs.shape[2]/100), dpi=100, frameon=False)
    #     fig.suptitle(attr)
    #     for i, idx in enumerate(vis_idxs):
    #         offset = idxs.index(idx)
    #         visualize_one(model, imgs[offset], masks[offset], labels[offset], patient_ids[offset], probs[offset], attr_names, axs[i])

    #     filename = 'vis_{}_step_{}.png'.format(attr.replace(' ', '_'), 0)
    #     plt.savefig(os.path.join(outdir, filename), dpi=100)
    #     plt.close()

    for idx, img in enumerate(imgs):
        visualize_one(model, idx, img, masks[idx], labels[idx], outdir)

# def visualize_one(model, img, mask, label, patient_id, prob, attr_names, axs):
def visualize_one(model, idx, img, mask, label, outdir):
    """ display [table of model vs ground truth probs | original image | grad-cam mask image] in a given suplot axs """
    # sort data by prob high to low
    # sort_idxs = prob.argsort()[::-1]
    # label = sort_idxs
    # prob = prob[sort_idxs]
    # names = [i for i in sort_idxs]
    # 1. left -- show table of ground truth and predictions, sorted by pred prob high to low
    # axs[0].set_title(patient_id)
    # data = np.stack([label, prob.round(3)]).T
    # axs[0].table(cellText=data, rowLabels=names, colLabels=['Ground truth', 'Pred. prob'],
    #              rowColours=plt.cm.Greens(0.5*label),
    #              cellColours=plt.cm.Greens(0.5*data), cellLoc='center', loc='center')
    # axs[0].axis('tight')
    # # 2. middle -- show original image
    # axs[1].set_title('Original image', fontsize=10)
    # axs[1].imshow(img.squeeze(), cmap='gray')
    # # 3. right -- show heatmap over original image with predictions
    # axs[2].set_title('Top class activation \n{}: {:.4f}'.format(names[0], prob[0]), fontsize=10)
    # axs[2].imshow(img.squeeze(), cmap='gray')
    # axs[2].imshow(mask.squeeze(), cmap='jet', alpha=0.5)

    # for ax in axs: ax.axis('off')
    fig, ax = plt.subplots(dpi=100, frameon=False)
    fig.suptitle(label)
    ax.imshow(img.squeeze(), cmap='gray')
    ax.imshow(mask.squeeze(), cmap='jet', alpha=0.5)
    ax.axis('off')

    filename = 'vis_{}.png'.format(idx, 0)
    plt.savefig(os.path.join(outdir, filename), dpi=100)
    plt.close()

def vis_attn(x, patient_ids, idxs, attn_layers, outdir, batch_element=0):
    H, W = x.shape[2:]
    nh = attn_layers[0].nh

    # select which pixels to visualize -- e.g. select virtices of a center square of side 1/3 of the image dims
    pix_to_vis = lambda h, w: [(h//3, w//3), (h//3, int(2*w/3)), (int(2*h/3), w//3), (int(2*h/3), int(2*w/3))]
    window = 30  # take mean attn around the pix_to_vis in a window of size ws

    for j, l in enumerate(attn_layers):
        # visualize attention maps (rows for each head; columns for each pixel)
        fig, axs = plt.subplots(nh+1, 4, figsize=(3,3/4*(1+nh)), frameon=False)
        fig.suptitle(patient_ids[batch_element], fontsize=8)
        # display target image; highlight pixel
        for ax, (ph, pw) in zip(axs[0], pix_to_vis(H,W)):
            image = x.clone().detach().mul_(0.0349).add_(0.5330)  # renormalize
            image[:,:,ph-window:ph+window,pw-window:pw+window] = torch.tensor([1., 215/255, 0]).view(1,3,1,1)   # add yellow pixel on the pix_to_vis for visualization
            ax.imshow(image[batch_element].permute(1,2,0).numpy())
            ax.axis('off')
        # display attention maps
        # get attention weights tensor for the batch element
        attn = l.weights.data[batch_element]
        # reshape attn tensor and select the pixels to visualize
        h = w = int(np.sqrt(attn.shape[-1]))
        ws = max(1, int(window * h/H))  # scale window to feature map size
        attn = attn.reshape(nh, h, w, h, w)
        for i, (ph, pw) in enumerate(pix_to_vis(h,w)):
            for h in range(nh):
                axs[h+1, i].imshow(attn[h, ph-ws:ph+ws, pw-ws:pw+ws, :, :].mean([0,1]).cpu().numpy())
                axs[h+1, i].axis('off')


        filename = 'attn_image_idx_{}_{}_layer_{}.png'.format(idxs[batch_element], batch_element, j)
        fig.subplots_adjust(0,0,1,0.95,0.05,0.05)
        plt.savefig(os.path.join(outdir, 'vis', filename))
        plt.close()

def plot_roc(metrics, outdir, filename, labels=ChexpertSmall.attr_names):
    fig, axs = plt.subplots(2, len(labels), figsize=(24,12))

    for i, (fpr, tpr, aucs, precision, recall, label) in enumerate(zip(metrics['fpr'].values(), metrics['tpr'].values(),
                                                                       metrics['aucs'].values(), metrics['precision'].values(),
                                                                       metrics['recall'].values(), labels)):
        # top row -- ROC
        axs[0,i].plot(fpr, tpr, label='AUC = %0.2f' % aucs)
        axs[0,i].plot([0, 1], [0, 1], 'k--')  # diagonal margin
        axs[0,i].set_xlabel('False Positive Rate')
        # bottom row - Precision-Recall
        axs[1,i].step(recall, precision, where='post')
        axs[1,i].set_xlabel('Recall')
        # format
        axs[0,i].set_title(label)
        axs[0,i].legend(loc="lower right")

    plt.suptitle(filename)
    axs[0,0].set_ylabel('True Positive Rate')
    axs[1,0].set_ylabel('Precision')

    for ax in axs.flatten():
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'plots', filename + '.png'), pad_inches=0.)
    plt.close()