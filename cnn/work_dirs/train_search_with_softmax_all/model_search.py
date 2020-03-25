import sys
sys.path.append('../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import numpy as np


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op) #14

  def forward(self, s0, s1, weights1,weights2,weights3,weights4):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
   
    sa=self._ops[0](s0,weights1[:8])+self._ops[1](s1,weights1[8:])

    sb=self._ops[2](s0,weights2[:8])+self._ops[3](s1,weights2[8:16])+self._ops[4](sa,weights2[16:])

    sc=self._ops[5](s0,weights3[:8])+self._ops[6](s1,weights3[8:16])+self._ops[7](sa,weights3[16:24])+self._ops[8](sb,weights3[24:])

    sd=self._ops[9](s0,weights4[:8])+self._ops[10](s1,weights4[8:16])+self._ops[11](sa,weights4[16:24])+self._ops[12](sb,weights4[24:32])+self._ops[13](sc,weights4[32:])
   
    states.append(sa)
    states.append(sb)
    states.append(sc)
    states.append(sd)

    # for i in range(self._steps): #

    #   s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
    #   offset += len(states)
    #   states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells): # 对每个cell
      if cell.reduction:
        weights1 = F.softmax(self.alphas_reduce1,dim=-1)
        weights2 = F.softmax(self.alphas_reduce2,dim=-1)
        weights3 = F.softmax(self.alphas_reduce3,dim=-1)
        weights4 = F.softmax(self.alphas_reduce4,dim=-1)
      else:
        weights1 = F.softmax(self.alphas_normal1,dim=-1)
        weights2 = F.softmax(self.alphas_normal2,dim=-1)
        weights3 = F.softmax(self.alphas_normal3,dim=-1)
        weights4 = F.softmax(self.alphas_normal4,dim=-1)

      s0, s1 = s1, cell(s0, s1, weights1,weights2,weights3,weights4)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    #k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    
    self.alphas_normal1 = Variable(1e-3*torch.randn(2*num_ops).cuda(), requires_grad=True)
    self.alphas_normal2 = Variable(1e-3*torch.randn(3*num_ops).cuda(), requires_grad=True)
    
    self.alphas_normal3 = Variable(1e-3*torch.randn(4*num_ops).cuda(), requires_grad=True)
    
    self.alphas_normal4 = Variable(1e-3*torch.randn(5*num_ops).cuda(), requires_grad=True)
    
    #self.alphas_normal=[self.alphas_normal1,self.alphas_normal2,self.alphas_normal3,]


    self.alphas_reduce1 = Variable(1e-3*torch.randn(2*num_ops).cuda(), requires_grad=True)
    self.alphas_reduce2 = Variable(1e-3*torch.randn(3*num_ops).cuda(), requires_grad=True)
    self.alphas_reduce3 = Variable(1e-3*torch.randn(4*num_ops).cuda(), requires_grad=True)
    self.alphas_reduce4 = Variable(1e-3*torch.randn(5*num_ops).cuda(), requires_grad=True)


    self._arch_parameters = [
      self.alphas_normal1,
      self.alphas_normal2,
      self.alphas_normal3,
      self.alphas_normal4,
      self.alphas_reduce1,
      self.alphas_reduce2,
      self.alphas_reduce3,
      self.alphas_reduce4
    ]

  def arch_parameters(self):
    return self._arch_parameters


  def genotype(self):

    def _parse(weights):
      gene = []
      for weight in weights:
        w = weight.copy()
        argsort_w=np.argsort(w)
        e= argsort_w[::-1]
        index=e[:2]
        for i in index:
          gene.append((PRIMITIVES[i%8], i//8))
      return gene

    rweights1 = F.softmax(self.alphas_reduce1,dim=-1).data.cpu().numpy()
    rweights2 = F.softmax(self.alphas_reduce2,dim=-1).data.cpu().numpy()
    rweights3 = F.softmax(self.alphas_reduce3,dim=-1).data.cpu().numpy()
    rweights4 = F.softmax(self.alphas_reduce4,dim=-1).data.cpu().numpy()

    nweights1 = F.softmax(self.alphas_normal1,dim=-1).data.cpu().numpy()
    nweights2 = F.softmax(self.alphas_normal2,dim=-1).data.cpu().numpy()
    nweights3 = F.softmax(self.alphas_normal3,dim=-1).data.cpu().numpy()
    nweights4 = F.softmax(self.alphas_normal4,dim=-1).data.cpu().numpy()

    gene_normal = _parse([nweights1,nweights2,nweights3,nweights4])
    gene_reduce = _parse([rweights1,rweights2,rweights3,rweights4])

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


