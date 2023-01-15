import torch
import torch.nn.functional
import time
from func import NewsDataset, ans_one_hot, pad_collate
from torch.utils.data import DataLoader


def train_model(vector, train_datasets, valid_datasets, batch_size, model, lossFn, optimizer, epoch_num, iter, device,
                embedding, epochs, sigopt, C_k):
    train_num, valid_num = iter[0], iter[1]
    train_batchsize, valid_batchsize = batch_size[0], batch_size[1]
    losses = []
    training_accuracies = []
    valid_accuracies = []
    valid_losses = []
    for epoch in range(epochs, epoch_num + 1):
        # training
        all_loss = 0.0
        training_TRUE = 0
        s_time = time.time()
        for aseqs, dseqs, ansss in train_datasets:
            ansss = ans_one_hot(ansss)
            amino_hoge, dna_hoge = vector.convert_vector(aseqs, dseqs)
            dataset_train = NewsDataset(amino_hoge, dna_hoge, ansss)
            train_dataset = DataLoader(dataset=dataset_train, batch_size=train_batchsize, drop_last=True, collate_fn=pad_collate)
            for aseq, dseq, ans in train_dataset:
                #aseq = aseq[:, :5]
                #dseq = dseq[:, :8]

                # クロスエントロピーを計算するように正解データを成形
                ans2 = ans.to(torch.float)
                anss = []
                ans = ans.numpy()
                for index in range(len(ans)):
                    if ans[index][0] == 1:
                        anss.append(0)
                    else:
                        anss.append(1)
                ans = torch.tensor(anss)

                # モデルが持ってる勾配の情報をリセット
                model.zero_grad()

                if device != 'cpu':
                    aseq = aseq.to(device)
                    dseq = dseq.to(device)
                    ans = ans.to(device)
                    ans2 = ans2.to(device)

                if embedding[0] == 'embedding':
                    aseq = model.amino_embed(aseq)
                    dseq = model.dna_embed(dseq)

                aminos, _ = model.amino_forward(aseq)
                dnas, _ = model.dna_forward(dseq)
                batch_sigmoid, _, _, _ = model.Integration(aminos, dnas)
                batch_loss = lossFn(batch_sigmoid, ans2)
                batch_loss.backward()
                optimizer.step()
                all_loss += batch_loss.item()
                _, pre = torch.max(batch_sigmoid, 1)
                n = torch.add(pre, ans)
                # 正解した個数
                TRUE = len(n[torch.where(n == 0)]) + len(n[torch.where(n == 2)])

                training_TRUE += TRUE
        all_loss = all_loss / train_num
        losses.append(all_loss)
        training_accuracy = training_TRUE / (train_num * train_batchsize)
        training_accuracies.append(training_accuracy)

        # validation
        print('start valid')
        valid_TRUE = 0
        valid_all_loss = 0.0
        with torch.no_grad():
            for aseqs, dseqs, ansss in valid_datasets:
                ansss = ans_one_hot(ansss)
                amino_hoge, dna_hoge = vector.convert_vector(aseqs, dseqs)
                dataset_train = NewsDataset(amino_hoge, dna_hoge, ansss)
                valid_dataset = DataLoader(dataset=dataset_train, batch_size=valid_batchsize, drop_last=True,
                                           collate_fn=pad_collate)
                for aseq, dseq, ans in valid_dataset:
                    # クロスエントロピーを計算するように正解データを成形
                    ans2 = ans.to(torch.float)
                    anss = []
                    ans = ans.numpy()
                    for index in range(len(ans)):
                        if ans[index][0] == 1:
                            anss.append(0)
                        else:
                            anss.append(1)
                    ans = torch.tensor(anss)

                    if device != 'cpu':
                        aseq = aseq.to(device)
                        dseq = dseq.to(device)
                        ans = ans.to(device)
                        ans2 = ans2.to(device)

                    if embedding[0] == 'embedding':
                        aseq = model.amino_embed(aseq)
                        dseq = model.dna_embed(dseq)

                    aminos, _ = model.amino_forward(aseq)
                    dnas, _ = model.dna_forward(dseq)
                    batch_sigmoid, _, _, _ = model.Integration(aminos, dnas)
                    batch_sigmoid = torch.reshape(batch_sigmoid, (-1, 2))
                    ans2 = torch.reshape(ans2, (-1, 2))
                    batch_loss = lossFn(batch_sigmoid, ans2)
                    valid_all_loss += batch_loss.item()
                    _, pre = torch.max(batch_sigmoid, 1)
                    n = torch.add(pre, ans)
                    # 正解した個数
                    TRUE = len(n[torch.where(n == 0)]) + len(n[torch.where(n == 2)])
                    valid_TRUE += TRUE
            valid_all_loss = valid_all_loss / valid_num
            valid_losses.append(valid_all_loss)
            valid_accuracy = valid_TRUE / (valid_num * valid_batchsize)
            valid_accuracies.append(valid_accuracy)

            e_time = time.time()
            write_list = [str(epoch), str(all_loss), str(training_accuracy), str(valid_all_loss), str(valid_accuracy)]
            f = open('result/' + str(C_k + 1) + 'result.txt', 'a')
            f.write('\t'.join(write_list))
            f.write('\n')
            f.close()

            print("epoch", epoch, "\t", "loss", valid_all_loss, "\t", "valid_accuracy", valid_accuracy, "\t",
                  "time", e_time - s_time)

            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint/' + str(C_k)
                       + 'checkpoint/checkpoint' + str(epoch) + '.pt')

            if sigopt == 'TRUE':
                if epoch > 4:
                    if valid_accuracies[epoch-1] < valid_accuracies[epoch-2] and valid_accuracies[epoch-2] < valid_accuracies[epoch-3] and valid_accuracies[epoch-3] < valid_accuracies[epoch-4]:
                        return losses, training_accuracies, valid_losses, valid_accuracies


    return losses, training_accuracies, valid_losses, valid_accuracies
