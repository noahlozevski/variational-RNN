import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from model import VRNN, Classifier
from tqdm import tqdm
from dataset import get_dataset as get_dataset2
import numpy as np
from processing import get_dataset
from collections import defaultdict
from sklearn.metrics import f1_score,precision_recall_fscore_support
txt_file = 'results.txt'
# added tqdm for progress monitoring


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


							
						
def main(runsss,overlap,window_size,h_dim,z_dim,batch_size,language):
	try:
		def train(epoch):
			train_loss = 0
			tq = tqdm(train_loader)
			for batch_idx, (data, _) in enumerate(tq):
				data = Variable(data.squeeze().transpose(0, 1))
				data = (data - data.min().item()) / (data.max().item() - data.min().item())
				#forward + backward + optimize
				optimizer.zero_grad()
				kld_loss, nll_loss, _, _ = model(data)
				loss = kld_loss + nll_loss
				loss.backward()
				optimizer.step()

				#grad norm clipping, only in pytorch version >= 1.10
				nn.utils.clip_grad_norm(model.parameters(), clip)

				tq.set_postfix(kld_loss=(kld_loss.item() / batch_size), nll_loss=(nll_loss.item() / batch_size))
				train_loss += loss.item()
			return
			
		def test(epoch):
			"""uses test data to evaluate 
			likelihood of the model"""
			
			mean_kld_loss, mean_nll_loss = 0, 0
			tq = tqdm(test_loader)
			for i, (data, _) in enumerate(tq):                                            
				
				#data = Variable(data)
				data = Variable(data.squeeze().transpose(0, 1))
				data = (data - data.min().item()) / (data.max().item() - data.min().item())

				kld_loss, nll_loss, _, _ = model(data)
			
				mean_kld_loss += kld_loss.item()
				mean_nll_loss += nll_loss.item()

			mean_kld_loss /= len(test_loader.dataset)
			mean_nll_loss /= len(test_loader.dataset)

			print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
				mean_kld_loss, mean_nll_loss))
			return
		
		def train_classifier(param_string, train_loader_enc, test_loader_enc, optimizer2, classify, criterion):
			num_epochs = 100
			best = 0
			acc = []
			train_acc = []
			for epoch in range(num_epochs):
				print(f'epoch num: {epoch}\n')
				running_loss = 0.0
				tq = tqdm(train_loader_enc)
				for i, (data, labels) in enumerate(tq):

					# zero the parameter gradients
					optimizer2.zero_grad()

					# forward + backward + optimize
					outputs = classify(data)
					# print(outputs, labels)
					loss = criterion(outputs.float(), labels.long())
					loss.backward()
					optimizer2.step()
					# print statistics
					running_loss += loss.item()
					tq.set_postfix(running_loss=(running_loss))
				acc.append(test_classifier(train_loader_enc,test_loader_enc,classify,flag=False,param_string=param_string))
				train_acc.append(test_classifier(train_loader_enc,test_loader_enc,classify,flag=True,param_string=param_string))
				print(acc[-1],best)
				if acc[-1] > best:
					best = acc[-1]
			print(f'best acc of {best}')
			f = open('class_acc.txt','a+')
			for i in range(len(train_acc)):
				f.write(param_string)
				f.write(f'{i},{train_acc[i]},{acc[i]}\n')
			f.close()
			return max([*acc, *train_acc])

		def test_classifier(train_loader_enc,test_loader_enc,classify,flag=False,param_string=""):
			# evaluate
			correct = 0
			total = 0
			classes = defaultdict(int)
			wrong = defaultdict(int)
			y_pred = []
			y_true = []
			with torch.no_grad():
				if flag:
					tq = tqdm(train_loader_enc)
					for data in tq:
						dat, labels = data
						outputs = classify(dat)
						for i in range(len(outputs)):
							if outputs[i][0] >= outputs[i][1]:
								pred = 0
							else:
								pred = 1
							y_pred.append(pred)
							y_true.append(labels[i].item())
							if pred == labels[i].item():
								correct += 1
								classes[pred] += 1
							else:
								wrong[pred] += 1
							total += 1
				else:
					tq = tqdm(test_loader_enc)
					for data in tq:
						dat, labels = data
						outputs = classify(dat)
						for i in range(len(outputs)):
							if outputs[i][0] >= outputs[i][1]:
								pred = 0
							else:
								pred = 1
							y_pred.append(pred)
							y_true.append(labels[i].item())
							if pred == labels[i].item():
								correct += 1
								classes[pred] += 1
							else:
								wrong[pred] += 1
							total += 1
			f11 = f1_score(y_true,y_pred,average='binary')
			[precision, recall, fbeta_score, support] = precision_recall_fscore_support(y_true,y_pred,average='binary')
			paramms = f'correct: {correct}, total:  {total}, classes:  {classes}, wrong: {wrong}, f1_score:  {f11}, precision: {precision}, recall: {recall}, fbeta_score: {fbeta_score}, support: {support}'
			print(paramms)
   
			print(f'accuracy: {correct/total}')
			if param_string:
				f = open('class_acc.txt','a+')
				f.write(param_string)
				f.write(f'y_pred-{y_pred},y_true-{y_true}\n')
				f.write(f'{paramms}\n')
				f.close()
			acc = correct/total
			return acc
 
		# transform inputs from test set to encoded vectors, make new training training loaders
		def transform_inputs(loader,batch_size=batch_size):
			encoded_inputs = []
			labels = []
			tq = tqdm(loader)
			with torch.no_grad():
				for batch_idx, (data, label) in enumerate(tq):
					data = Variable(data.squeeze().transpose(0, 1))
					data = (data - data.min().item()) / (data.max().item() - data.min().item())
					h = model.predict(data)
					for i in range(h.shape[1]):
						encoded_inputs.append(h[:,i,:].flatten().numpy())
						labels.append(label[i].item())
			return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(encoded_inputs), torch.Tensor(labels)), batch_size=batch_size,shuffle=True)
 
		def run_classifier(epochh,fn):
			for b_size in [4,8,16,32]:
				train_loader_enc = transform_inputs(train_loader, b_size)
				test_loader_enc = transform_inputs(test_loader, b_size)
				for intermediate_dim in [5,10,20,40]:
					for layers in [True, False]:
						f = open(txt_file,'a+')
						f.write(f'fn="{fn}",runsss={runsss},epochh={epochh},overlap={overlap},window_size={window_size},h_dim={h_dim},z_dim={z_dim},batch_size={batch_size},language={language},b_size={b_size},intermediate_dim={intermediate_dim},layers={layers}\n')
						f.close()
					
						classify = Classifier(input_dim=h_dim, intermediate_dim=20,layers=layers)
						print(classify, classify.count_parameters())
						criterion = nn.CrossEntropyLoss()
						optimizer2 = torch.optim.Adam(classify.parameters(), lr=0.001)


						train_classifier(f'fn="{fn}",runsss={runsss},epochh={epochh},overlap={overlap},window_size={window_size},h_dim={h_dim},z_dim={z_dim},batch_size={batch_size},language={language},b_size={b_size},intermediate_dim={intermediate_dim},layers={layers}\n',
                       train_loader_enc, test_loader_enc, optimizer2, classify, criterion)
						accuracy = test_classifier(train_loader_enc,test_loader_enc,classify,flag=False)
				
						print(f'final accuracy ---> {accuracy}')
						print('Finished Training')
			return
		x = get_dataset2(overlap=overlap, window_size=window_size, time_steps=20, language=language, max_len=(100 if language=="english" else 80))
		#hyperparameters
		# x_dim = 70
		x_dim = x['genuine'].shape[2]
		# h_dim = 100
		# z_dim = 16
		n_layers =  1
		n_epochs = 25
		clip = 10
		learning_rate = 3e-4
		batch_size = batch_size
		seed = 128
		print_every = 10
		save_every = 5


		#manual seed
		torch.manual_seed(seed)

		#init model + optimizer + datasets
		x_i = []
		y_i = []
		x_it = []
		y_it = []
		test_count = defaultdict(int)
		tot = 80
		for val in ['genuine','forged']:
			for i in x[val]:
				if val == 'genuine' and test_count['genuine'] < tot:
					x_it.append(i)
					y_it.append([1])
					test_count['genuine'] += 1
				elif val == 'forged' and test_count['forged'] < tot:
					x_it.append(i)
					y_it.append([0])
					test_count['forged'] += 1
				else:
					x_i.append(i)
					y_i.append([0] if val == 'genuine' else [1])
				
		# print(len(x_i),len(y_i),len(x_it),len(y_it))
		x_i,y_i,x_it,y_it = np.array(x_i),np.array(y_i).reshape((-1,)),np.array(x_it),np.array(y_it).reshape((-1,))

		if False:
			signatures_train, signatures_test, labels_train, labels_test = get_dataset()
		else:
			signatures_train, signatures_test, labels_train, labels_test = x_i, x_it, y_i, y_it

		print('input data\n', signatures_train.shape, labels_train.shape, signatures_test.shape, labels_test.shape)

		x_dim = signatures_train.shape[2]

		train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(signatures_train),
																																							torch.Tensor(labels_train)),
																							batch_size=batch_size,
																							shuffle=True)
		test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(signatures_test),
																																						torch.Tensor(labels_test)),
																							batch_size=batch_size,
																							shuffle=True)

		model = VRNN(x_dim, h_dim, z_dim, n_layers)

		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

		for epoch in range(1, n_epochs + 1):
			
			#training + testing
			train(epoch)
			test(epoch)

			#saving model
			if epoch % save_every == 1:
				fn = 'saves/vrnn_state_dict_'+f'{runsss},{overlap},{window_size},{h_dim},{z_dim},{batch_size}.{language},{epoch}'+'.pth'
				torch.save(model.state_dict(), fn)
				print('Saved model to ' + fn)
				run_classifier(epoch,fn+'\n')

		# freeze model weights
		for param in model.parameters():
			param.requires_grad = False
				

		
		
  	
	except:
		print('FAILED RUN')
		f = open(txt_file,'a+')
		f.write(f'FAILED RUN ---> {runsss},{overlap},{window_size},{h_dim},{z_dim},{batch_size}.{language}\n')
		f.close()

runsss = 0
for overlap in [2,4,5,7]:
	for window_size in [3,5,7,10,15]:
		if window_size / overlap < .5:
			continue
		for h_dim in [20, 40, 60, 80, 100]:
			for z_dim in [8, 16, 24, 48, 60]:
				for batch_size in [16, 32, 64, 128]:
					for language in ['chinese','english']:
						# f = open(txt_file,'a+')
						# f.write(f'{runsss},{overlap},{window_size},{h_dim},{z_dim},{batch_size}.{language}')
						# f.close()
						main(runsss,overlap,window_size,h_dim,z_dim,batch_size,language)

						runsss += 1





# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=True, download=True,
# 		transform=transforms.ToTensor()),
    # batch_size=batch_size, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=False, 
# 		transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True)
