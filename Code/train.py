import torch
import time
import copy
from helper import group_eval

from load import log

from operators import *

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, outdir, logfile):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    PATH = outdir
    global_epoch = num_epochs

    for epoch in range(num_epochs):
        log(logfile, ("Epoch: %f/%f" % (epoch, num_epochs - 1)))
        log(logfile, ("----------"))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
        
            running_loss = 0.0
            running_corrects = 0

            for batch in dataloaders[phase]:
                length = len(batch)
                if length == 2:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                elif length == 3:
                    inputs, labels, g = batch
                    inputs, labels, g = inputs.to(device), labels.to(device), g.to(device)
                else:
                    inputs, labels, g, ID, sp = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        outputs= model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        loss.backward()
                        optimizer.step()

                    if phase == 'val':
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == 'val':
                scheduler.step(epoch_loss)
            
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            log(logfile, ('%s Loss: %f Acc: %f') % (phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        global_epoch -= 1

        if epoch % 10 == 0:
            PATH = outdir + 'model_' + str(epoch)
            torch.save({
            'epoch': global_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best val acc': best_acc,
            }, PATH)

        log(logfile, ('----------'))
 
 
    time_elapsed = time.time() - since
    log(logfile, ("Training complete in %sm %ss" % (time_elapsed // 60, time_elapsed % 60)))
    log(logfile, ("Best val Acc: %s" % best_acc))

    model.load_state_dict(best_model_wts)
    PATH = outdir + 'model_' + str(global_epoch)
    torch.save({
            'epoch': global_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best val acc': best_acc,
            }, PATH)
    return model, val_acc_history


def test_model(dataloader, trainmodel, device, criterion, config, logfile):
    size = len(dataloader.dataset)
    trainmodel.eval()
    test_loss, correct = 0, 0
    test_loss_group, correct_group = {}, {}
    group_size = {}

    if config['adversary'] == True:
        test_loss_adv, correct_adv = 0, 0
        diversity_adv = 0

    #test operator
    if config['rotate'] == True:
        test_loss_rotate, correct_rotate = 0, 0
        test_loss_center_rotate, correct_center_rotate = 0, 0
        test_loss_outer_rotate, correct_outer_rotate = 0, 0
        diversity_rotate = 0
        diversity_center_rotate = 0
        diversity_outer_rotate = 0
        #rotate
        operator_rotate = Patch_Embed_Rotate(
            img_size=config['image_size'], patch_size=config['patch_size'], in_chans=3, p=config['rotate_percent']
        )
        operator_rotate.to(device)
        #center_rotate
        operator_center_rotate = Patch_Embed_Center_Rotate(
            img_size=config['image_size'], patch_size=config['patch_size'], in_chans=3
        )
        operator_center_rotate.to(device)
        #outer_rotate
        operator_outer_rotate = Patch_Embed_Outer_Rotate(
            img_size=config['image_size'], patch_size=config['patch_size'], in_chans=3
        )
        operator_outer_rotate.to(device)

    if config['blur'] == True:
        test_loss_blur, correct_blur = 0, 0
        diversity_blur = 0
        #blur
        operator_blur = Patch_Embed_Blur(
            img_size=config['image_size'], patch_size=config['patch_size'], h1=config['h1'], h2=config['h2']
        )
        operator_blur.to(device)

    if config['shuffle'] == True:
        test_loss_shuffle, correct_shuffle = 0, 0
        diversity_shuffle = 0
        #shuffle
        operator_shuffle = Patch_Embed_Shuffle(
            img_size=config['image_size'], patch_size=config['patch_size'], in_chans=3, groups=config['shuffle_groups']
        )
        operator_shuffle.to(device)

    if config['occlude'] == True:
        test_loss_occlude, correct_occlude = 0, 0
        test_loss_center_occlude, correct_center_occlude = 0, 0
        test_loss_outer_occlude, correct_outer_occlude = 0, 0
        diversity_occlude = 0
        diversity_center_occlude = 0
        diversity_outer_occlude = 0
        #occlude
        operator_occlude = Patch_Embed_Occlude(
            img_size=config['image_size'], patch_size=config['patch_size'], in_chans=3, p=config['occlude_percent']
        )
        operator_occlude.to(device)
        #center_occlude
        operator_center_occlude = Patch_Embed_Center_Occlude(
            img_size=config['image_size'], patch_size=config['patch_size'], in_chans=3
        )
        operator_center_occlude.to(device)
        #outer_occlude
        operator_outer_occlude = Patch_Embed_Outer_Occlude(
            img_size=config['image_size'], patch_size=config['patch_size'], in_chans=3
        )
        operator_outer_occlude.to(device)

    # with torch.no_grad():
    for ibatch, batch in enumerate(dataloader):
        length = len(batch)
        if length == 2:
            X, y = batch
            X, y = X.to(device), y.to(device)
        elif length == 3:
            X, y, g = batch
            # inputs = group_eval(X, y, g)
            # # print(inputs.keys())
            # for key in inputs.keys():
            #     if key not in test_loss_group:
            #         test_loss_group[key] = 0
            #         correct_group[key] = 0
            #         group_size[key] = 0
            #     X_temp, y_temp = inputs[key]
            #     X_temp = torch.stack(X_temp)
            #     y_temp = torch.stack(y_temp)
            #     X_temp, y_temp = X_temp.to(device), y_temp.to(device)
            #     group_size[key] += X_temp.size(0)
            #     outputs_temp = trainmodel(X_temp)
            #     loss_temp = criterion(outputs_temp, y_temp)
            #     _, preds_temp = torch.max(outputs_temp, 1)

            #     test_loss_group[key] += loss_temp.item() * X_temp.size(0)
            #     correct_group[key] += torch.sum(preds_temp == y_temp.data)
            X, y, g = X.to(device), y.to(device), g.to(device)
        else:
            X, y, g, ID, sp = batch
            inputs = group_eval(X, y, g)
            # print(inputs.keys())
            for key in inputs.keys():
                if key not in test_loss_group:
                    test_loss_group[key] = 0
                    correct_group[key] = 0
                    group_size[key] = 0
                X_temp, y_temp = inputs[key]
                X_temp = torch.stack(X_temp)
                y_temp = torch.stack(y_temp)
                X_temp, y_temp = X_temp.to(device), y_temp.to(device)
                group_size[key] += X_temp.size(0)
                outputs_temp = trainmodel(X_temp)
                loss_temp = criterion(outputs_temp, y_temp)
                _, preds_temp = torch.max(outputs_temp, 1)

                test_loss_group[key] += loss_temp.item() * X_temp.size(0)
                correct_group[key] += torch.sum(preds_temp == y_temp.data)
            X, y, g = X.to(device), y.to(device), g.to(device)

        outputs = trainmodel(X)
        loss = criterion(outputs, y)
        _, preds = torch.max(outputs, 1)

        test_loss += loss.item() * X.size(0)
        correct += torch.sum(preds == y.data)

        if config['adversary'] == True:
            #adversary
            X_adv = pgd_attack(trainmodel, X, y, device)
            outputs_adv = trainmodel(X_adv)
            loss_adv = criterion(outputs_adv, y)
            _, preds_adv = torch.max(outputs_adv, 1)

            test_loss_adv += loss_adv.item() * X_adv.size(0)
            correct_adv += torch.sum(preds_adv == y.data)

            diversity_adv += torch.sum(preds != preds_adv)

        with torch.no_grad():
            if config['rotate'] == True:
                #rotate
                X_rotate = operator_rotate(X)
                outputs_rotate = trainmodel(X_rotate)
                loss_rotate = criterion(outputs_rotate, y)
                _, preds_rotate = torch.max(outputs_rotate, 1)

                test_loss_rotate += loss_rotate.item() * X_rotate.size(0)
                correct_rotate += torch.sum(preds_rotate == y.data)

                diversity_rotate += torch.sum(preds != preds_rotate)

                #center rotate
                X_crotate = operator_center_rotate(X)
                outputs_crotate = trainmodel(X_crotate)
                loss_crotate = criterion(outputs_crotate, y)
                _, preds_crotate = torch.max(outputs_crotate, 1)

                test_loss_center_rotate += loss_crotate.item() * X_crotate.size(0)
                correct_center_rotate += torch.sum(preds_crotate == y.data)

                diversity_center_rotate += torch.sum(preds != preds_crotate)

                #outer rotate
                X_outrotate = operator_outer_rotate(X)
                outputs_outrotate = trainmodel(X_outrotate)
                loss_outrotate = criterion(outputs_outrotate, y)
                _, preds_outrotate = torch.max(outputs_outrotate, 1)

                test_loss_outer_rotate += loss_outrotate.item() * X_outrotate.size(0)
                correct_outer_rotate += torch.sum(preds_outrotate == y.data)

                diversity_outer_rotate += torch.sum(preds != preds_outrotate)

            if config['blur'] == True:
                #blur
                X_blur = operator_blur(X)
                outputs_blur = trainmodel(X_blur)
                loss_blur = criterion(outputs_blur, y)
                _, preds_blur = torch.max(outputs_blur, 1)

                test_loss_blur += loss_blur.item() * X_blur.size(0)
                correct_blur += torch.sum(preds_blur == y.data)

                diversity_blur += torch.sum(preds != preds_blur)

            if config['shuffle'] == True:
                #shuffle
                X_shuffle = operator_shuffle(X)
                outputs_shuffle = trainmodel(X_shuffle)
                loss_shuffle = criterion(outputs_shuffle, y)
                _, preds_shuffle = torch.max(outputs_shuffle, 1)

                test_loss_shuffle += loss_shuffle.item() * X_shuffle.size(0)
                correct_shuffle += torch.sum(preds_shuffle != y.data)

                diversity_shuffle += torch.sum(preds != preds_shuffle)

            if config['occlude'] == True:
                #occlude
                X_occlude = operator_occlude(X)
                outputs_occlude = trainmodel(X_occlude)
                loss_occlude = criterion(outputs_occlude, y)
                _, preds_occlude = torch.max(outputs_occlude, 1)

                test_loss_occlude += loss_occlude.item() * X_occlude.size(0)
                correct_occlude += torch.sum(preds_occlude == y.data)

                diversity_occlude += torch.sum(preds != preds_occlude)

                #center occlude
                X_cocclude = operator_center_occlude(X)
                outputs_cocclude = trainmodel(X_cocclude)
                loss_cocclude = criterion(outputs_cocclude, y)
                _, preds_cocclude = torch.max(outputs_cocclude, 1)

                test_loss_center_occlude += loss_cocclude.item() * X_cocclude.size(0)
                correct_center_occlude += torch.sum(preds_cocclude != y.data)

                diversity_center_occlude += torch.sum(preds != preds_cocclude)

                #outer occlude
                X_outerocclude = operator_outer_occlude(X)
                outputs_outerocclude = trainmodel(X_outerocclude)
                loss_outerocclude = criterion(outputs_outerocclude, y)
                _, preds_outerocclude = torch.max(outputs_outerocclude, 1)

                test_loss_outer_occlude += loss_outerocclude.item() * X_outerocclude.size(0)
                correct_outer_occlude += torch.sum(preds_outerocclude == y.data)

                diversity_outer_occlude += torch.sum(preds != preds_outerocclude)

        if ibatch % 100 == 0:
            log(logfile, ("current: %s" % (ibatch * len(X))))

    correct = correct.double()
    if config['adversary'] == True:
        correct_adv = correct_adv.double()
        diversity_adv = diversity_adv.double()
        test_loss_adv /= size
        correct_adv /= size
        diversity_adv /= size
        log(logfile, (f"Pass rate for adversarial examples: {(100*correct_adv):>0.1f}%, Avg loss after adversarial examples: {test_loss_adv:>8f} \n \
            Diversity of adversarial examples: {(100*diversity_adv):>0.1f}% \n"))
    if config['rotate'] == True:
        correct_rotate = correct_rotate.double()
        correct_center_rotate = correct_center_rotate.double()
        correct_outer_rotate = correct_outer_rotate.double()
        diversity_rotate = diversity_rotate.double()
        diversity_center_rotate = diversity_center_rotate.double()
        diversity_outer_rotate = diversity_outer_rotate.double()
        test_loss_rotate /= size
        correct_rotate /= size
        test_loss_center_rotate /= size
        correct_center_rotate /= size
        test_loss_outer_rotate /= size
        correct_outer_rotate /= size
        diversity_rotate /= size
        diversity_center_rotate /= size
        diversity_outer_rotate /= size
        log(logfile, (f"Pass rate for rotation: {(100*correct_rotate):>0.1f}%, Avg loss after rotation: {test_loss_rotate:>8f} \n \
            Diversity of rotation: {(100*diversity_rotate):>0.1f}% \n"))
        log(logfile, (f"Pass rate for center rotation: {(100*correct_center_rotate):>0.1f}%, Avg loss after center rotation: {test_loss_center_rotate:>8f} \n \
                Diversity of center rotation: {(100*diversity_center_rotate):>0.1f}% \n"))
        log(logfile, (f"Pass rate for outer rotation: {(100*correct_outer_rotate):>0.1f}%, Avg loss after outer rotation: {test_loss_outer_rotate:>8f} \n \
                Diversity of outer rotation: {(100*diversity_outer_rotate):>0.1f}% \n"))
    if config['blur'] == True:
        correct_blur = correct_blur.double()
        diversity_blur = diversity_blur.double()
        test_loss_blur /= size
        correct_blur /= size
        diversity_blur /= size
        log(logfile, (f"Pass rate for blur: {(100*correct_blur):>0.1f}%, Avg loss after blur: {test_loss_blur:>8f} \n \
            Diversity of blur: {(100*diversity_blur):>0.1f}% \n"))
    if config['shuffle'] == True:
        correct_shuffle = correct_shuffle.double()
        diversity_shuffle = diversity_shuffle.double()
        test_loss_shuffle /= size
        correct_shuffle /= size
        diversity_shuffle /= size
        log(logfile, (f"Pass rate for shuffle: {(100*correct_shuffle):>0.1f}%, Avg loss after shuffle: {test_loss_shuffle:>8f} \n \
            Diversity of shuffle: {(100*diversity_shuffle):>0.1f}% \n"))
    if config['occlude'] == True:
        correct_occlude = correct_occlude.double()
        correct_center_occlude = correct_center_occlude.double()
        correct_outer_occlude = correct_outer_occlude.double()
        diversity_occlude = diversity_occlude.double()
        diversity_center_occlude = diversity_center_occlude.double()
        diversity_outer_occlude = diversity_outer_occlude.double()
        test_loss_occlude /= size
        correct_occlude /= size
        test_loss_center_occlude /= size
        correct_center_occlude /= size
        test_loss_outer_occlude /= size
        correct_outer_occlude /= size
        diversity_occlude /= size
        diversity_center_occlude /= size
        diversity_outer_occlude /= size
        log(logfile, (f"Pass rate for occlude: {(100*correct_occlude):>0.1f}%, Avg loss after occlude: {test_loss_occlude:>8f} \n \
            Diversity of occlude: {(100*diversity_occlude):>0.1f}% \n"))
        log(logfile, (f"Pass rate for center occlude: {(100*correct_center_occlude):>0.1f}%, Avg loss after center occlude: {test_loss_center_occlude:>8f} \n \
                Diversity of center occlude: {(100*diversity_center_occlude):>0.1f}% \n"))
        log(logfile, (f"Pass rate for outer occlude: {(100*correct_outer_occlude):>0.1f}%, Avg loss after outer occlude: {test_loss_outer_occlude:>8f} \n \
                Diversity of outer occlude: {(100*diversity_outer_occlude):>0.1f}% \n"))

    test_loss /= size
    correct /= size
    log(logfile, (f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"))
    
    for key in test_loss_group.keys():
        test_loss_g = test_loss_group[key]
        correct_g = correct_group[key].double()
        test_loss_g /= group_size[key]
        correct_g /= group_size[key]
        log(logfile, (f"Test Error Group {key}: \n Accuracy: {(100*correct_g):>0.1f}%, Avg loss: {test_loss_g:>8f} \n"))

def test_model_imga(dataloader, trainmodel, device, criterion, config, logfile):
    size = len(dataloader.dataset)
    trainmodel.eval()
    test_loss, correct = 0, 0
    test_loss_group, correct_group = {}, {}
    group_size = {}

    thousand_to_200 = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: 1, 7: -1, 8: -1, 9: -1, 10: -1, 11: 1, 12: -1, 13: 1, 14: -1, 15: 1, 16: -1, 17: 1, 18: -1, 19: -1, 20: -1, 21: -1, 22: 1, 23: 1, 24: -1, 25: -1, 26: -1, 27: 1, 28: -1, 29: -1, 30: 1, 31: -1, 32: -1, 33: -1, 34: -1, 35: -1, 36: -1, 37: 1, 38: -1, 39: 1, 40: -1, 41: -1, 42: 1, 43: -1, 44: -1, 45: -1, 46: -1, 47: 1, 48: -1, 49: -1, 50: 1, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: 1, 58: -1, 59: -1, 60: -1, 61: -1, 62: -1, 63: -1, 64: -1, 65: -1, 66: -1, 67: -1, 68: -1, 69: -1, 70: 1, 71: 1, 72: -1, 73: -1, 74: -1, 75: -1, 76: 1, 77: -1, 78: -1, 79: 1, 80: -1, 81: -1, 82: -1, 83: -1, 84: -1, 85: -1, 86: -1, 87: -1, 88: -1, 89: 1, 90: 1, 91: -1, 92: -1, 93: -1, 94: 1, 95: -1, 96: 1, 97: 1, 98: -1, 99: 1, 100: -1, 101: -1, 102: -1, 103: -1, 104: -1, 105: 1, 106: -1, 107: 1, 108: 1, 109: -1, 110: 1, 111: -1, 112: -1, 113: 1, 114: -1, 115: -1, 116: -1, 117: -1, 118: -1, 119: -1, 120: -1, 121: -1, 122: -1, 123: -1, 124: 1, 125: 1, 126: -1, 127: -1, 128: -1, 129: -1, 130: 1, 131: -1, 132: 1, 133: -1, 134: -1, 135: -1, 136: -1, 137: -1, 138: -1, 139: -1, 140: -1, 141: -1, 142: -1, 143: 1, 144: 1, 145: -1, 146: -1, 147: -1, 148: -1, 149: -1, 150: 1, 151: 1, 152: -1, 153: -1, 154: -1, 155: -1, 156: -1, 157: -1, 158: -1, 159: -1, 160: -1, 161: -1, 162: -1, 163: -1, 164: -1, 165: -1, 166: -1, 167: -1, 168: -1, 169: -1, 170: -1, 171: -1, 172: -1, 173: -1, 174: -1, 175: -1, 176: -1, 177: -1, 178: -1, 179: -1, 180: -1, 181: -1, 182: -1, 183: -1, 184: -1, 185: -1, 186: -1, 187: -1, 188: -1, 189: -1, 190: -1, 191: -1, 192: -1, 193: -1, 194: -1, 195: -1, 196: -1, 197: -1, 198: -1, 199: -1, 200: -1, 201: -1, 202: -1, 203: -1, 204: -1, 205: -1, 206: -1, 207: 1, 208: -1, 209: -1, 210: -1, 211: -1, 212: -1, 213: -1, 214: -1, 215: -1, 216: -1, 217: -1, 218: -1, 219: -1, 220: -1, 221: -1, 222: -1, 223: -1, 224: -1, 225: -1, 226: -1, 227: -1, 228: -1, 229: -1, 230: -1, 231: -1, 232: -1, 233: -1, 234: 1, 235: 1, 236: -1, 237: -1, 238: -1, 239: -1, 240: -1, 241: -1, 242: -1, 243: -1, 244: -1, 245: -1, 246: -1, 247: -1, 248: -1, 249: -1, 250: -1, 251: -1, 252: -1, 253: -1, 254: 1, 255: -1, 256: -1, 257: -1, 258: -1, 259: -1, 260: -1, 261: -1, 262: -1, 263: -1, 264: -1, 265: -1, 266: -1, 267: -1, 268: -1, 269: -1, 270: -1, 271: -1, 272: -1, 273: -1, 274: -1, 275: -1, 276: -1, 277: 1, 278: -1, 279: -1, 280: -1, 281: -1, 282: -1, 283: 1, 284: -1, 285: -1, 286: -1, 287: 1, 288: -1, 289: -1, 290: -1, 291: 1, 292: -1, 293: -1, 294: -1, 295: 1, 296: -1, 297: -1, 298: 1, 299: -1, 300: -1, 301: 1, 302: -1, 303: -1, 304: -1, 305: -1, 306: 1, 307: 1, 308: 1, 309: 1, 310: 1, 311: 1, 312: -1, 313: 1, 314: 1, 315: 1, 316: -1, 317: 1, 318: -1, 319: 1, 320: -1, 321: -1, 322: -1, 323: 1, 324: 1, 325: -1, 326: 1, 327: 1, 328: -1, 329: -1, 330: 1, 331: -1, 332: -1, 333: -1, 334: 1, 335: 1, 336: 1, 337: -1, 338: -1, 339: -1, 340: -1, 341: -1, 342: -1, 343: -1, 344: -1, 345: -1, 346: -1, 347: 1, 348: -1, 349: -1, 350: -1, 351: -1, 352: -1, 353: -1, 354: -1, 355: -1, 356: -1, 357: -1, 358: -1, 359: -1, 360: -1, 361: 1, 362: -1, 363: 1, 364: -1, 365: -1, 366: -1, 367: -1, 368: -1, 369: -1, 370: -1, 371: -1, 372: 1, 373: -1, 374: -1, 375: -1, 376: -1, 377: -1, 378: 1, 379: -1, 380: -1, 381: -1, 382: -1, 383: -1, 384: -1, 385: -1, 386: 1, 387: -1, 388: -1, 389: -1, 390: -1, 391: -1, 392: -1, 393: -1, 394: -1, 395: -1, 396: -1, 397: 1, 398: -1, 399: -1, 400: 1, 401: 1, 402: 1, 403: -1, 404: 1, 405: -1, 406: -1, 407: 1, 408: -1, 409: -1, 410: -1, 411: 1, 412: -1, 413: -1, 414: -1, 415: -1, 416: 1, 417: 1, 418: -1, 419: -1, 420: 1, 421: -1, 422: -1, 423: -1, 424: -1, 425: 1, 426: -1, 427: -1, 428: 1, 429: -1, 430: 1, 431: -1, 432: -1, 433: -1, 434: -1, 435: -1, 436: -1, 437: 1, 438: 1, 439: -1, 440: -1, 441: -1, 442: -1, 443: -1, 444: -1, 445: 1, 446: -1, 447: -1, 448: -1, 449: -1, 450: -1, 451: -1, 452: -1, 453: -1, 454: -1, 455: -1, 456: 1, 457: 1, 458: -1, 459: -1, 460: -1, 461: 1, 462: 1, 463: -1, 464: -1, 465: -1, 466: -1, 467: -1, 468: -1, 469: -1, 470: 1, 471: -1, 472: 1, 473: -1, 474: -1, 475: -1, 476: -1, 477: -1, 478: -1, 479: -1, 480: -1, 481: -1, 482: -1, 483: 1, 484: -1, 485: -1, 486: 1, 487: -1, 488: 1, 489: -1, 490: -1, 491: -1, 492: 1, 493: -1, 494: -1, 495: -1, 496: 1, 497: -1, 498: -1, 499: -1, 500: -1, 501: -1, 502: -1, 503: -1, 504: -1, 505: -1, 506: -1, 507: -1, 508: -1, 509: -1, 510: -1, 511: -1, 512: -1, 513: -1, 514: 1, 515: -1, 516: 1, 517: -1, 518: -1, 519: -1, 520: -1, 521: -1, 522: -1, 523: -1, 524: -1, 525: -1, 526: -1, 527: -1, 528: 1, 529: -1, 530: 1, 531: -1, 532: -1, 533: -1, 534: -1, 535: -1, 536: -1, 537: -1, 538: -1, 539: 1, 540: -1, 541: -1, 542: 1, 543: 1, 544: -1, 545: -1, 546: -1, 547: -1, 548: -1, 549: 1, 550: -1, 551: -1, 552: 1, 553: -1, 554: -1, 555: -1, 556: -1, 557: 1, 558: -1, 559: -1, 560: -1, 561: 1, 562: 1, 563: -1, 564: -1, 565: -1, 566: -1, 567: -1, 568: -1, 569: 1, 570: -1, 571: -1, 572: 1, 573: 1, 574: -1, 575: 1, 576: -1, 577: -1, 578: -1, 579: 1, 580: -1, 581: -1, 582: -1, 583: -1, 584: -1, 585: -1, 586: -1, 587: -1, 588: -1, 589: 1, 590: -1, 591: -1, 592: -1, 593: -1, 594: -1, 595: -1, 596: -1, 597: -1, 598: -1, 599: -1, 600: -1, 601: -1, 602: -1, 603: -1, 604: -1, 605: -1, 606: 1, 607: 1, 608: -1, 609: 1, 610: -1, 611: -1, 612: -1, 613: -1, 614: 1, 615: -1, 616: -1, 617: -1, 618: -1, 619: -1, 620: -1, 621: -1, 622: -1, 623: -1, 624: -1, 625: -1, 626: 1, 627: 1, 628: -1, 629: -1, 630: -1, 631: -1, 632: -1, 633: -1, 634: -1, 635: -1, 636: -1, 637: -1, 638: -1, 639: -1, 640: 1, 641: 1, 642: 1, 643: 1, 644: -1, 645: -1, 646: -1, 647: -1, 648: -1, 649: -1, 650: -1, 651: -1, 652: -1, 653: -1, 654: -1, 655: -1, 656: -1, 657: -1, 658: 1, 659: -1, 660: -1, 661: -1, 662: -1, 663: -1, 664: -1, 665: -1, 666: -1, 667: -1, 668: 1, 669: -1, 670: -1, 671: -1, 672: -1, 673: -1, 674: -1, 675: -1, 676: -1, 677: 1, 678: -1, 679: -1, 680: -1, 681: -1, 682: 1, 683: -1, 684: 1, 685: -1, 686: -1, 687: 1, 688: -1, 689: -1, 690: -1, 691: -1, 692: -1, 693: -1, 694: -1, 695: -1, 696: -1, 697: -1, 698: -1, 699: -1, 700: -1, 701: 1, 702: -1, 703: -1, 704: 1, 705: -1, 706: -1, 707: -1, 708: -1, 709: -1, 710: -1, 711: -1, 712: -1, 713: -1, 714: -1, 715: -1, 716: -1, 717: -1, 718: -1, 719: 1, 720: -1, 721: -1, 722: -1, 723: -1, 724: -1, 725: -1, 726: -1, 727: -1, 728: -1, 729: -1, 730: -1, 731: -1, 732: -1, 733: -1, 734: -1, 735: -1, 736: 1, 737: -1, 738: -1, 739: -1, 740: -1, 741: -1, 742: -1, 743: -1, 744: -1, 745: -1, 746: 1, 747: -1, 748: -1, 749: 1, 750: -1, 751: -1, 752: 1, 753: -1, 754: -1, 755: -1, 756: -1, 757: -1, 758: 1, 759: -1, 760: -1, 761: -1, 762: -1, 763: 1, 764: -1, 765: 1, 766: -1, 767: -1, 768: 1, 769: -1, 770: -1, 771: -1, 772: -1, 773: 1, 774: 1, 775: -1, 776: 1, 777: -1, 778: -1, 779: 1, 780: 1, 781: -1, 782: -1, 783: -1, 784: -1, 785: -1, 786: 1, 787: -1, 788: -1, 789: -1, 790: -1, 791: -1, 792: 1, 793: -1, 794: -1, 795: -1, 796: -1, 797: 1, 798: -1, 799: -1, 800: -1, 801: -1, 802: 1, 803: 1, 804: 1, 805: -1, 806: -1, 807: -1, 808: -1, 809: -1, 810: -1, 811: -1, 812: -1, 813: 1, 814: -1, 815: 1, 816: -1, 817: -1, 818: -1, 819: -1, 820: 1, 821: -1, 822: -1, 823: 1, 824: -1, 825: -1, 826: -1, 827: -1, 828: -1, 829: -1, 830: -1, 831: 1, 832: -1, 833: 1, 834: -1, 835: 1, 836: -1, 837: -1, 838: -1, 839: 1, 840: -1, 841: -1, 842: -1, 843: -1, 844: -1, 845: 1, 846: -1, 847: 1, 848: -1, 849: -1, 850: 1, 851: -1, 852: -1, 853: -1, 854: -1, 855: -1, 856: -1, 857: -1, 858: -1, 859: 1, 860: -1, 861: -1, 862: 1, 863: -1, 864: -1, 865: -1, 866: -1, 867: -1, 868: -1, 869: -1, 870: 1, 871: -1, 872: -1, 873: -1, 874: -1, 875: -1, 876: -1, 877: -1, 878: -1, 879: 1, 880: 1, 881: -1, 882: -1, 883: -1, 884: -1, 885: -1, 886: -1, 887: -1, 888: 1, 889: -1, 890: 1, 891: -1, 892: -1, 893: -1, 894: -1, 895: -1, 896: -1, 897: 1, 898: -1, 899: -1, 900: 1, 901: -1, 902: -1, 903: -1, 904: -1, 905: -1, 906: -1, 907: 1, 908: -1, 909: -1, 910: -1, 911: -1, 912: -1, 913: 1, 914: -1, 915: -1, 916: -1, 917: -1, 918: -1, 919: -1, 920: -1, 921: -1, 922: -1, 923: -1, 924: 1, 925: -1, 926: -1, 927: -1, 928: -1, 929: -1, 930: -1, 931: -1, 932: 1, 933: 1, 934: 1, 935: -1, 936: -1, 937: 1, 938: -1, 939: -1, 940: -1, 941: -1, 942: -1, 943: 1, 944: -1, 945: 1, 946: -1, 947: 1, 948: -1, 949: -1, 950: -1, 951: 1, 952: -1, 953: -1, 954: 1, 955: -1, 956: 1, 957: 1, 958: -1, 959: 1, 960: -1, 961: -1, 962: -1, 963: -1, 964: -1, 965: -1, 966: -1, 967: -1, 968: -1, 969: -1, 970: -1, 971: 1, 972: 1, 973: -1, 974: -1, 975: -1, 976: -1, 977: -1, 978: -1, 979: -1, 980: 1, 981: 1, 982: -1, 983: -1, 984: 1, 985: -1, 986: 1, 987: 1, 988: 1, 989: -1, 990: -1, 991: -1, 992: -1, 993: -1, 994: -1, 995: -1, 996: -1, 997: -1, 998: -1, 999: -1}

    indices_in_1k = [k for k in thousand_to_200 if thousand_to_200[k] != -1]

    for ibatch, batch in enumerate(dataloader):
        length = len(batch)
        if length == 2:
            X, y = batch
            X, y = X.to(device), y.to(device)
        elif length == 3:
            X, y, g = batch
            X, y, g = X.to(device), y.to(device), g.to(device)
        else:
            X, y, g, ID, sp = batch
            inputs = group_eval(X, y, g)
            # print(inputs.keys())
            for key in inputs.keys():
                if key not in test_loss_group:
                    test_loss_group[key] = 0
                    correct_group[key] = 0
                    group_size[key] = 0
                X_temp, y_temp = inputs[key]
                X_temp = torch.stack(X_temp)
                y_temp = torch.stack(y_temp)
                X_temp, y_temp = X_temp.to(device), y_temp.to(device)
                group_size[key] += X_temp.size(0)
                outputs_temp = trainmodel(X_temp)
                loss_temp = criterion(outputs_temp, y_temp)
                _, preds_temp = torch.max(outputs_temp, 1)

                test_loss_group[key] += loss_temp.item() * X_temp.size(0)
                correct_group[key] += torch.sum(preds_temp == y_temp.data)
            X, y, g = X.to(device), y.to(device), g.to(device)

        outputs = trainmodel(X)[:,indices_in_1k]
        loss = criterion(outputs, y)
        _, preds = torch.max(outputs, 1)

        test_loss += loss.item() * X.size(0)
        correct += torch.sum(preds == y.data)
    
    correct = correct.double()

    test_loss /= size
    correct /= size
    log(logfile, (f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"))

def test_model_imgr(dataloader, trainmodel, device, criterion, config, logfile):
    size = len(dataloader.dataset)
    trainmodel.eval()
    test_loss, correct = 0, 0
    test_loss_group, correct_group = {}, {}
    group_size = {}

    thousand_to_200 = {0: -1, 1: 1, 2: 1, 3: -1, 4: 1, 5: -1, 6: 1, 7: -1, 8: 1, 9: 1, 10: -1, 11: 1, 12: -1, 13: 1, 14: -1, 15: -1, 16: -1, 17: -1, 18: -1, 19: -1, 20: -1, 21: -1, 22: 1, 23: 1, 24: -1, 25: -1, 26: 1, 27: -1, 28: -1, 29: 1, 30: -1, 31: 1, 32: -1, 33: -1, 34: -1, 35: -1, 36: -1, 37: -1, 38: -1, 39: 1, 40: -1, 41: -1, 42: -1, 43: -1, 44: -1, 45: -1, 46: -1, 47: 1, 48: -1, 49: -1, 50: -1, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: -1, 58: -1, 59: -1, 60: -1, 61: -1, 62: -1, 63: 1, 64: -1, 65: -1, 66: -1, 67: -1, 68: -1, 69: -1, 70: -1, 71: 1, 72: -1, 73: -1, 74: -1, 75: -1, 76: 1, 77: -1, 78: -1, 79: 1, 80: -1, 81: -1, 82: -1, 83: -1, 84: 1, 85: -1, 86: -1, 87: -1, 88: -1, 89: -1, 90: 1, 91: -1, 92: -1, 93: -1, 94: 1, 95: -1, 96: 1, 97: 1, 98: -1, 99: 1, 100: 1, 101: -1, 102: -1, 103: -1, 104: -1, 105: 1, 106: -1, 107: 1, 108: -1, 109: -1, 110: -1, 111: -1, 112: -1, 113: 1, 114: -1, 115: -1, 116: -1, 117: -1, 118: -1, 119: -1, 120: -1, 121: -1, 122: 1, 123: -1, 124: -1, 125: 1, 126: -1, 127: -1, 128: -1, 129: -1, 130: 1, 131: -1, 132: 1, 133: -1, 134: -1, 135: -1, 136: -1, 137: -1, 138: -1, 139: -1, 140: -1, 141: -1, 142: -1, 143: -1, 144: 1, 145: 1, 146: -1, 147: 1, 148: 1, 149: -1, 150: 1, 151: 1, 152: -1, 153: -1, 154: -1, 155: 1, 156: -1, 157: -1, 158: -1, 159: -1, 160: 1, 161: 1, 162: 1, 163: 1, 164: -1, 165: -1, 166: -1, 167: -1, 168: -1, 169: -1, 170: -1, 171: 1, 172: 1, 173: -1, 174: -1, 175: -1, 176: -1, 177: -1, 178: 1, 179: -1, 180: -1, 181: -1, 182: -1, 183: -1, 184: -1, 185: -1, 186: -1, 187: 1, 188: -1, 189: -1, 190: -1, 191: -1, 192: -1, 193: -1, 194: -1, 195: 1, 196: -1, 197: -1, 198: -1, 199: 1, 200: -1, 201: -1, 202: -1, 203: 1, 204: -1, 205: -1, 206: -1, 207: 1, 208: 1, 209: -1, 210: -1, 211: -1, 212: -1, 213: -1, 214: -1, 215: -1, 216: -1, 217: -1, 218: -1, 219: 1, 220: -1, 221: -1, 222: -1, 223: -1, 224: -1, 225: -1, 226: -1, 227: -1, 228: -1, 229: -1, 230: -1, 231: 1, 232: 1, 233: -1, 234: 1, 235: 1, 236: -1, 237: -1, 238: -1, 239: -1, 240: -1, 241: -1, 242: 1, 243: -1, 244: -1, 245: 1, 246: -1, 247: 1, 248: -1, 249: -1, 250: 1, 251: 1, 252: -1, 253: -1, 254: 1, 255: -1, 256: -1, 257: -1, 258: -1, 259: 1, 260: 1, 261: -1, 262: -1, 263: 1, 264: -1, 265: 1, 266: -1, 267: 1, 268: -1, 269: 1, 270: -1, 271: -1, 272: -1, 273: -1, 274: -1, 275: -1, 276: 1, 277: 1, 278: -1, 279: -1, 280: -1, 281: 1, 282: -1, 283: -1, 284: -1, 285: -1, 286: -1, 287: -1, 288: 1, 289: 1, 290: -1, 291: 1, 292: 1, 293: 1, 294: -1, 295: -1, 296: 1, 297: -1, 298: -1, 299: 1, 300: -1, 301: 1, 302: -1, 303: -1, 304: -1, 305: -1, 306: -1, 307: -1, 308: 1, 309: 1, 310: 1, 311: 1, 312: -1, 313: -1, 314: 1, 315: 1, 316: -1, 317: -1, 318: -1, 319: 1, 320: -1, 321: -1, 322: -1, 323: 1, 324: -1, 325: -1, 326: -1, 327: 1, 328: -1, 329: -1, 330: 1, 331: -1, 332: -1, 333: -1, 334: 1, 335: 1, 336: -1, 337: 1, 338: 1, 339: -1, 340: 1, 341: 1, 342: -1, 343: -1, 344: 1, 345: -1, 346: -1, 347: 1, 348: -1, 349: -1, 350: -1, 351: -1, 352: -1, 353: 1, 354: -1, 355: 1, 356: -1, 357: -1, 358: -1, 359: -1, 360: -1, 361: 1, 362: 1, 363: -1, 364: -1, 365: 1, 366: 1, 367: 1, 368: 1, 369: -1, 370: -1, 371: -1, 372: 1, 373: -1, 374: -1, 375: -1, 376: -1, 377: -1, 378: -1, 379: -1, 380: -1, 381: -1, 382: -1, 383: -1, 384: -1, 385: -1, 386: -1, 387: -1, 388: 1, 389: -1, 390: 1, 391: -1, 392: -1, 393: 1, 394: -1, 395: -1, 396: -1, 397: 1, 398: -1, 399: -1, 400: -1, 401: 1, 402: -1, 403: -1, 404: -1, 405: -1, 406: -1, 407: 1, 408: -1, 409: -1, 410: -1, 411: -1, 412: -1, 413: 1, 414: 1, 415: -1, 416: -1, 417: -1, 418: -1, 419: -1, 420: -1, 421: -1, 422: -1, 423: -1, 424: -1, 425: 1, 426: -1, 427: -1, 428: 1, 429: -1, 430: 1, 431: -1, 432: -1, 433: -1, 434: -1, 435: 1, 436: -1, 437: 1, 438: -1, 439: -1, 440: -1, 441: 1, 442: -1, 443: -1, 444: -1, 445: -1, 446: -1, 447: 1, 448: 1, 449: -1, 450: -1, 451: -1, 452: -1, 453: -1, 454: -1, 455: -1, 456: -1, 457: 1, 458: -1, 459: -1, 460: -1, 461: -1, 462: 1, 463: 1, 464: -1, 465: -1, 466: -1, 467: -1, 468: -1, 469: 1, 470: 1, 471: 1, 472: 1, 473: -1, 474: -1, 475: -1, 476: 1, 477: -1, 478: -1, 479: -1, 480: -1, 481: -1, 482: -1, 483: 1, 484: -1, 485: -1, 486: -1, 487: 1, 488: -1, 489: -1, 490: -1, 491: -1, 492: -1, 493: -1, 494: -1, 495: -1, 496: -1, 497: -1, 498: -1, 499: -1, 500: -1, 501: -1, 502: -1, 503: -1, 504: -1, 505: -1, 506: -1, 507: -1, 508: -1, 509: -1, 510: -1, 511: -1, 512: -1, 513: -1, 514: -1, 515: 1, 516: -1, 517: -1, 518: -1, 519: -1, 520: -1, 521: -1, 522: -1, 523: -1, 524: -1, 525: -1, 526: -1, 527: -1, 528: -1, 529: -1, 530: -1, 531: -1, 532: -1, 533: -1, 534: -1, 535: -1, 536: -1, 537: -1, 538: -1, 539: -1, 540: -1, 541: -1, 542: -1, 543: -1, 544: -1, 545: -1, 546: 1, 547: -1, 548: -1, 549: -1, 550: -1, 551: -1, 552: -1, 553: -1, 554: -1, 555: 1, 556: -1, 557: -1, 558: 1, 559: -1, 560: -1, 561: -1, 562: -1, 563: -1, 564: -1, 565: -1, 566: -1, 567: -1, 568: -1, 569: -1, 570: 1, 571: -1, 572: -1, 573: -1, 574: -1, 575: -1, 576: -1, 577: -1, 578: -1, 579: 1, 580: -1, 581: -1, 582: -1, 583: 1, 584: -1, 585: -1, 586: -1, 587: 1, 588: -1, 589: -1, 590: -1, 591: -1, 592: -1, 593: 1, 594: 1, 595: -1, 596: 1, 597: -1, 598: -1, 599: -1, 600: -1, 601: -1, 602: -1, 603: -1, 604: -1, 605: -1, 606: -1, 607: -1, 608: -1, 609: 1, 610: -1, 611: -1, 612: -1, 613: 1, 614: -1, 615: -1, 616: -1, 617: 1, 618: -1, 619: -1, 620: -1, 621: 1, 622: -1, 623: -1, 624: -1, 625: -1, 626: -1, 627: -1, 628: -1, 629: 1, 630: -1, 631: -1, 632: -1, 633: -1, 634: -1, 635: -1, 636: -1, 637: 1, 638: -1, 639: -1, 640: -1, 641: -1, 642: -1, 643: -1, 644: -1, 645: -1, 646: -1, 647: -1, 648: -1, 649: -1, 650: -1, 651: -1, 652: -1, 653: -1, 654: -1, 655: -1, 656: -1, 657: 1, 658: 1, 659: -1, 660: -1, 661: -1, 662: -1, 663: -1, 664: -1, 665: -1, 666: -1, 667: -1, 668: -1, 669: -1, 670: -1, 671: -1, 672: -1, 673: -1, 674: -1, 675: -1, 676: -1, 677: -1, 678: -1, 679: -1, 680: -1, 681: -1, 682: -1, 683: -1, 684: -1, 685: -1, 686: -1, 687: -1, 688: -1, 689: -1, 690: -1, 691: -1, 692: -1, 693: -1, 694: -1, 695: -1, 696: -1, 697: -1, 698: -1, 699: -1, 700: -1, 701: 1, 702: -1, 703: -1, 704: -1, 705: -1, 706: -1, 707: -1, 708: -1, 709: -1, 710: -1, 711: -1, 712: -1, 713: -1, 714: -1, 715: -1, 716: -1, 717: 1, 718: -1, 719: -1, 720: -1, 721: -1, 722: -1, 723: -1, 724: 1, 725: -1, 726: -1, 727: -1, 728: -1, 729: -1, 730: -1, 731: -1, 732: -1, 733: -1, 734: -1, 735: -1, 736: -1, 737: -1, 738: -1, 739: -1, 740: -1, 741: -1, 742: -1, 743: -1, 744: -1, 745: -1, 746: -1, 747: -1, 748: -1, 749: -1, 750: -1, 751: -1, 752: -1, 753: -1, 754: -1, 755: -1, 756: -1, 757: -1, 758: -1, 759: -1, 760: -1, 761: -1, 762: -1, 763: 1, 764: -1, 765: -1, 766: -1, 767: -1, 768: 1, 769: -1, 770: -1, 771: -1, 772: -1, 773: -1, 774: 1, 775: -1, 776: 1, 777: -1, 778: -1, 779: 1, 780: 1, 781: -1, 782: -1, 783: -1, 784: -1, 785: -1, 786: -1, 787: 1, 788: -1, 789: -1, 790: -1, 791: -1, 792: -1, 793: -1, 794: -1, 795: -1, 796: -1, 797: -1, 798: -1, 799: -1, 800: -1, 801: -1, 802: -1, 803: -1, 804: -1, 805: 1, 806: -1, 807: -1, 808: -1, 809: -1, 810: -1, 811: -1, 812: 1, 813: -1, 814: -1, 815: 1, 816: -1, 817: -1, 818: -1, 819: -1, 820: 1, 821: -1, 822: -1, 823: -1, 824: 1, 825: -1, 826: -1, 827: -1, 828: -1, 829: -1, 830: -1, 831: -1, 832: -1, 833: 1, 834: -1, 835: -1, 836: -1, 837: -1, 838: -1, 839: -1, 840: -1, 841: -1, 842: -1, 843: -1, 844: -1, 845: -1, 846: -1, 847: 1, 848: -1, 849: -1, 850: -1, 851: -1, 852: 1, 853: -1, 854: -1, 855: -1, 856: -1, 857: -1, 858: -1, 859: -1, 860: -1, 861: -1, 862: -1, 863: -1, 864: -1, 865: -1, 866: 1, 867: -1, 868: -1, 869: -1, 870: -1, 871: -1, 872: -1, 873: -1, 874: -1, 875: 1, 876: -1, 877: -1, 878: -1, 879: -1, 880: -1, 881: -1, 882: -1, 883: 1, 884: -1, 885: -1, 886: -1, 887: -1, 888: -1, 889: 1, 890: -1, 891: -1, 892: -1, 893: -1, 894: -1, 895: 1, 896: -1, 897: -1, 898: -1, 899: -1, 900: -1, 901: -1, 902: -1, 903: -1, 904: -1, 905: -1, 906: -1, 907: 1, 908: -1, 909: -1, 910: -1, 911: -1, 912: -1, 913: -1, 914: -1, 915: -1, 916: -1, 917: -1, 918: -1, 919: -1, 920: -1, 921: -1, 922: -1, 923: -1, 924: -1, 925: -1, 926: -1, 927: -1, 928: 1, 929: -1, 930: -1, 931: 1, 932: 1, 933: 1, 934: 1, 935: -1, 936: 1, 937: 1, 938: -1, 939: -1, 940: -1, 941: -1, 942: -1, 943: 1, 944: -1, 945: 1, 946: -1, 947: 1, 948: 1, 949: 1, 950: -1, 951: 1, 952: -1, 953: 1, 954: 1, 955: -1, 956: -1, 957: 1, 958: -1, 959: -1, 960: -1, 961: -1, 962: -1, 963: 1, 964: -1, 965: 1, 966: -1, 967: 1, 968: -1, 969: -1, 970: -1, 971: -1, 972: -1, 973: -1, 974: -1, 975: -1, 976: -1, 977: -1, 978: -1, 979: -1, 980: 1, 981: 1, 982: -1, 983: 1, 984: -1, 985: -1, 986: -1, 987: -1, 988: 1, 989: -1, 990: -1, 991: -1, 992: -1, 993: -1, 994: -1, 995: -1, 996: -1, 997: -1, 998: -1, 999: -1}

    indices_in_1k = [k for k in thousand_to_200 if thousand_to_200[k] != -1]

    for ibatch, batch in enumerate(dataloader):
        length = len(batch)
        if length == 2:
            X, y = batch
            X, y = X.to(device), y.to(device)
        elif length == 3:
            X, y, g = batch
            X, y, g = X.to(device), y.to(device), g.to(device)
        else:
            X, y, g, ID, sp = batch
            inputs = group_eval(X, y, g)
            # print(inputs.keys())
            for key in inputs.keys():
                if key not in test_loss_group:
                    test_loss_group[key] = 0
                    correct_group[key] = 0
                    group_size[key] = 0
                X_temp, y_temp = inputs[key]
                X_temp = torch.stack(X_temp)
                y_temp = torch.stack(y_temp)
                X_temp, y_temp = X_temp.to(device), y_temp.to(device)
                group_size[key] += X_temp.size(0)
                outputs_temp = trainmodel(X_temp)
                loss_temp = criterion(outputs_temp, y_temp)
                _, preds_temp = torch.max(outputs_temp, 1)

                test_loss_group[key] += loss_temp.item() * X_temp.size(0)
                correct_group[key] += torch.sum(preds_temp == y_temp.data)
            X, y, g = X.to(device), y.to(device), g.to(device)

        outputs = trainmodel(X)[:,indices_in_1k]
        loss = criterion(outputs, y)
        _, preds = torch.max(outputs, 1)

        test_loss += loss.item() * X.size(0)
        correct += torch.sum(preds == y.data)
    
    correct = correct.double()

    test_loss /= size
    correct /= size
    log(logfile, (f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"))


def test_model_imgc(dataloader, trainmodel, device, criterion, config, logfile):
    test_loss, correct = 0, 0
    test_loss_group, correct_group = {}, {}
    group_size = {}
    trainmodel.eval()
    size = 0
    for c in dataloader.keys():
        for loader in dataloader[c]:
            load_size = len(loader.dataset)
            size += load_size
            for ibatch, batch in enumerate(loader):
                length = len(batch)
                if length == 2:
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                elif length == 3:
                    X, y, g = batch
                    X, y, g = X.to(device), y.to(device), g.to(device)
                else:
                    X, y, g, ID, sp = batch
                    inputs = group_eval(X, y, g)
                    # print(inputs.keys())
                    for key in inputs.keys():
                        if key not in test_loss_group:
                            test_loss_group[key] = 0
                            correct_group[key] = 0
                            group_size[key] = 0
                        X_temp, y_temp = inputs[key]
                        X_temp = torch.stack(X_temp)
                        y_temp = torch.stack(y_temp)
                        X_temp, y_temp = X_temp.to(device), y_temp.to(device)
                        group_size[key] += X_temp.size(0)
                        outputs_temp = trainmodel(X_temp)
                        loss_temp = criterion(outputs_temp, y_temp)
                        _, preds_temp = torch.max(outputs_temp, 1)

                        test_loss_group[key] += loss_temp.item() * X_temp.size(0)
                        correct_group[key] += torch.sum(preds_temp == y_temp.data)
                    X, y, g = X.to(device), y.to(device), g.to(device)

                outputs = trainmodel(X)
                loss = criterion(outputs, y)
                _, preds = torch.max(outputs, 1)

                test_loss += loss.item() * X.size(0)
                correct += torch.sum(preds == y.data)
            
    correct = correct.double()

    test_loss /= size
    correct /= size
    log(logfile, (f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"))

def test_model_counterfactual(dataloader, dataset, trainmodel, device, criterion, config, logfile):
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    Tensor = FloatTensor

    tv_beta = 3
    learning_rate = 0.1
    max_iterations = 500
    l1_coeff = 0.01
    tv_coeff = 0.2

    size = len(dataloader.dataset)
    trainmodel.eval()
    test_loss, correct = 0, 0
    test_loss_group, correct_group = {}, {}
    group_size = {}

    for ibatch, batch in enumerate(dataset):
        X, y, g = batch
        X, y, g = X.to(device), y.to(device), g.to(device)

        outputs = trainmodel(X)
        loss = criterion(outputs, y)
        _, preds = torch.max(outputs, 1)

        test_loss += loss.item() * X.size(0)
        correct += torch.sum(preds == y.data)

    