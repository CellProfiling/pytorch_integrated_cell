import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

def dragan_penalty(X, discriminator, gpu_id):
    #untested
    lambda_ = 10
    batch_size = X.size(0)

    # gradient penalty
    alpha = torch.rand(batch_size, 1).expand(X.size()).cuda(gpu_id)
    alpha2 = torch.rand(batch_size, 1).expand(X.size()).cuda(gpu_id)

    pdb.set_trace()

    x_hat = Variable(alpha * X.data + (1 - alpha) * (X.data + 0.5 * X.data.std() * alpha2), requires_grad=True)
    pred_hat = discriminator(x_hat)
    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(gpu_id), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = lambda_ * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    gradient_penalty.backward()

def iteration(enc, dec, encD, decD,
              optEnc, optDec, optEncD, optDecD,
              critRecon, critZClass, critZRef, critEncD, critDecD,
              dataProvider, opt):
    gpu_id = opt.gpu_ids[0]

    ###update the discriminator
    #maximize log(AdvZ(z)) + log(1 - AdvZ(Enc(x)))
    for p in encD.parameters(): # reset requires_grad
        p.requires_grad = True # they are set to False below in netG update

    for p in decD.parameters():
        p.requires_grad = True

    for p in enc.parameters():
        p.requires_grad = False

    for p in dec.parameters():
        p.requires_grad = False

    rand_inds_encD = np.random.permutation(opt.ndat)
    niter = len(range(0, len(rand_inds_encD), opt.batch_size))
    inds_encD = (rand_inds_encD[i:i+opt.batch_size] for i in range(0, len(rand_inds_encD), opt.batch_size))

    inds = next(inds_encD)
    x = Variable(dataProvider.get_images(inds,'train')).cuda(gpu_id)

    if opt.nClasses > 0:
        classes = Variable(dataProvider.get_classes(inds,'train')).cuda(gpu_id)

    if opt.nRef > 0:
        ref = Variable(dataProvider.get_ref(inds,'train')).cuda(gpu_id)

    zAll = enc(x)

    for var in zAll:
        var.detach_()

    xHat = dec(zAll)

    zReal = Variable(opt.latentSample(opt.batch_size, opt.nlatentdim)).cuda(gpu_id)
    zFake = zAll[-1] # why just the last image?

    optEnc.zero_grad()
    optDec.zero_grad()
    optEncD.zero_grad()
    optDecD.zero_grad()

    ### train encD
    y_zReal = Variable(torch.ones(opt.batch_size)).cuda(gpu_id)
    y_zFake = Variable(torch.zeros(opt.batch_size)).cuda(gpu_id)
    # is this training or initialization?

    # train with real
    yHat_zReal = encD(zReal)
    errEncD_real = critEncD(yHat_zReal, y_zReal)

    # train with fake
    yHat_zFake = encD(zFake)
    errEncD_fake = critEncD(yHat_zFake, y_zFake)

    encDLoss = (errEncD_real + errEncD_fake)/2
    encDLoss.backward(retain_variables=True)

    if opt.dragan:
        dragan_penalty(zReal, encD, gpu_id) #what does this do in place?

    optEncD.step()

    ###Train decD
    if opt.nClasses > 0:
        y_xReal = classes
        y_xFake = Variable(torch.LongTensor(opt.batch_size).fill_(opt.nClasses)).cuda(gpu_id)
    else:
        y_xReal = Variable(torch.ones(opt.batch_size)).cuda(gpu_id)
        y_xFake = Variable(torch.zeros(opt.batch_size)).cuda(gpu_id)

    yHat_xReal = decD(x)
    errDecD_real = critDecD(yHat_xReal, y_xReal)

    #train with fake, reconstructed
    yHat_xFake = decD(xHat)
    errDecD_fake = critDecD(yHat_xFake, y_xFake)

    #train with fake, sampled and decoded
    zAll[-1] = zReal

    yHat_xFake2 = decD(dec(zAll))
    errEncD_fake2 = critDecD(yHat_xFake2, y_xFake)

    decDLoss = (errDecD_real + (errDecD_fake + errEncD_fake2)/2)/2
    decDLoss.backward(retain_variables=True)

    if opt.dragan:
        dragan_penalty(x, decD, gpu_id)

    optDecD.step()

    for p in enc.parameters():
        p.requires_grad = True

    for p in dec.parameters():
        p.requires_grad = True

    for p in encD.parameters():
        p.requires_grad = False

    for p in decD.parameters():
        p.requires_grad = False

    optEnc.zero_grad()
    optDec.zero_grad()
    optEncD.zero_grad()
    optDecD.zero_grad()

    ## train the autoencoder
    zAll = enc(x)
    xHat = dec(zAll)

    c = 0
    if opt.nClasses > 0:
        classLoss = critZClass(zAll[c], classes)
        c += 1
    else:
        classLoss = Variable(torch.zeros(1)).cuda(gpu_id)

    if opt.nRef > 0:
        refLoss = critZRef(zAll[c], ref)
        c += 1
    else:
        refLoss = Variable(torch.zeros(1)).cuda(gpu_id)

    reconLoss = critRecon(xHat, x)

    #update wrt encD
    yHatFake = encD(zAll[c])
    minimaxEncDLoss = critEncD(yHatFake, y_zReal)

    # total encoder loss for single backprop call per model component
    totEncLoss = classLoss + refLoss + reconLoss + minimaxEncDLoss.mul(opt.encDRatio)
    totEncLoss.backward(retain_variables=True)

    optEnc.step()

    for p in enc.parameters():
        p.requires_grad = False

    #update wrt decD(dec(enc(X)))
    yHat_xFake = decD(xHat)
    minimaxDecDLoss = critDecD(yHat_xFake, y_xReal)

    #update wrt decD(dec(Z))
    zAll[c] = Variable(opt.latentSample(opt.batch_size, opt.nlatentdim)).cuda(gpu_id)
    xHat = dec(zAll)

    yHat_xFake2 = decD(xHat)
    minimaxDecDLoss2 = critDecD(yHat_xFake2, y_xReal)

    minimaxDecLoss = (minimaxDecDLoss+minimaxDecDLoss2)/2

    # total DecD loss for single backprop call per model component
    totminimaxDecLoss = minimaxDecLoss.mul(opt.decDRatio)
    totminimaxDecLoss.backward(retain_variables=True)

    optDec.step()


    errors = (reconLoss.data[0],)
    if opt.nClasses > 0:
        errors += (classLoss.data[0],)

    if opt.nRef > 0:
        errors += (refLoss.data[0],)

    errors += (minimaxEncDLoss.data[0], encDLoss.data[0], minimaxDecLoss.data[0], decDLoss.data[0])

    return errors, zFake.data
