import torch
import torch.nn.functional
from func import NewsDataset, ans_one_hot, attention_view, pad_collate
from torch.utils.data import DataLoader


def test_model(vector, test_datasets, batch_size, model, test_num, device, test_tf, test_ta, view, embedding, dir, att,
               stride, mer, amino_stride, amino_mer, map, epoch, Occulusion, test):

    with torch.no_grad():
        test_TRUE = 0
        TP, FP, FN, TN = 0, 0, 0, 0
        TPs, FPs, FNs, TNs = [], [], [], []
        num = 0
        score = []
        for aseqs, dseqs, ansss in test_datasets:
            ansss = ans_one_hot(ansss)
            amino_hoge, dna_hoge = vector.convert_vector(aseqs, dseqs)
            if Occulusion:
                a = ansss
                for i in range(len(amino_hoge)):
                    ansss = torch.cat([ansss, a])
                batch_size = len(amino_hoge)
            dataset_test = NewsDataset(amino_hoge, dna_hoge, ansss)
            test_dataset = DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=True, collate_fn=pad_collate)
            for aseq, dseq, ans in test_dataset:
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

                if embedding[0] == 'embedding':
                    aseq = model.amino_embed(aseq)
                    dseq = model.dna_embed(dseq)
                aminos, amino_att = model.amino_forward(aseq)
                dnas, dna_att = model.dna_forward(dseq)

                if att:
                    batch_sigmoid, amino_att, dna_attmap, dna_att = model.Integration(aminos, dnas)
                else:
                    batch_sigmoid, _, _, _ = model.Integration(aminos, dnas)
                    dna_attmap = 0

                batch_sigmoid = torch.reshape(batch_sigmoid, (-1, 2))
                _, pre = torch.max(batch_sigmoid, 1)

                for i in range(len(pre)):
                    score.append(batch_sigmoid[i, 1].item())
                    N = num * batch_size + i
                    if pre[i] == 1 and ans[i] == 1:
                        TPs.append(num * batch_size + i)
                        TP += 1
                        if view == 'TRUE':
                            attention_view(dir, test_tf, test_ta, num, batch_size, 'TP', i, ans, aseqs, dseqs, pre,
                                           amino_att, dna_att, embedding, amino_stride, stride, amino_mer, mer, map,
                                           vector, dna_attmap, N)

                    elif pre[i] == 1 and ans[i] == 0:
                        FPs.append(num * batch_size + i)
                        FP += 1
                        if view == 'TRUE':
                            attention_view(dir, test_tf, test_ta, num, batch_size, 'FP', i, ans, aseqs, dseqs, pre,
                                           amino_att, dna_att, embedding, amino_stride, stride, amino_mer, mer, map,
                                           vector, dna_attmap, N)

                    elif pre[i] == 0 and ans[i] == 1:
                        FNs.append(num * batch_size + i)
                        FN += 1
                        if view == 'TRUE':
                            attention_view(dir, test_tf, test_ta, num, batch_size, 'FN', i, ans, aseqs, dseqs, pre,
                                           amino_att, dna_att, embedding, amino_stride, stride, amino_mer, mer, map,
                                           vector, dna_attmap, N)

                    else:
                        TNs.append(num * batch_size + i)
                        TN += 1
                        if view == 'TRUE':
                            attention_view(dir, test_tf, test_ta, num, batch_size, 'TN', i, ans, aseqs, dseqs, pre,
                                           amino_att, dna_att, embedding, amino_stride, stride, amino_mer, mer, map,
                                           vector, dna_attmap, N)

                n = torch.add(pre, ans)
                # 正解した個数
                TRUE = len(n[torch.where(n == 0)]) + len(n[torch.where(n == 2)])
                test_TRUE += TRUE
                num += 1

        if Occulusion:
            cat_attn = vector.view_Occulusion(score, test['tf_seq'][0], test['ta_seq'][0], test_tf, test_ta, ans, dir, pre)

        try:
            Precision = TP / (TP + FP)
        except:
            Precision = 0
        try:
            Recall = TP / (TP + FN)
        except:
            Recall = 0
        try:
            F_measure = 2 * Recall * Precision / (Recall + Precision)
        except:
            F_measure = 0

        test_accuracy = test_TRUE / test_num
        print("test_accuracy", test_accuracy)
        print("TP", TP, "\t", "FP", FP, "\t", "FN", FN, "\t", "TN", TN, "\t", "Recall", Recall, "\t", "Precision", Precision, "\t", "F_measure", F_measure)
        write_list = [str(epoch), str(test_accuracy), str(TP), str(FP), str(FN), str(TN), str(Recall), str(Precision), str(F_measure)]
        if Occulusion:
            return cat_attn
        if view == "FALSE":
            f = open(dir + 'Details/testresult.txt', 'a')
            f.write('\t'.join(write_list))
            f.write('\n')
            f.close()

        return TPs, FPs, FNs, TNs, score

