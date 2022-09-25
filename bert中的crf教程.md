# 1. 损失函数
CRF是对序列加约束的常见方式，其训练目标是让`golden序列`在所有序列组合中的概率最大
- 发射概率： $E^t_{j}$ 表示t时刻映射到`tag j`的非归一化概率
- 转移概率： $T_{ij}$ 表示`tag i`转移到`tag j`的概率
- 输入序列 X，输出序列 y
- 序列y的分数： $$S(X, y)=\sum_{t=0}^n{T_{y_t,y_{t+1}}} + \sum_{t=0}^n{E_{y_t}^t}$$
- 序列y的似然概率： $Y_x$ 表示所有的tag序列 $$P(y|X)=\frac{e^{S(X, y)}}{\sum_{\check{y}\in{Y_x}}^n{e^{S(X, \check{y})}}}$$
- Loss表示: 前一项表示所有tag的得分和，后一项表示真实序列tag的得分 $$Loss=-log(P(y|X))=-log(\frac{e^{S(X, y)}}{\sum_{\check{y}\in{Y_x}}{e^{S(X, \check{y})}}})=log(\sum_{\check{y}\in{Y_x}}{e^{S(X, \check{y})}})-S(X, y)$$

# 2. 递推公式
t时刻以 $y_t$ 结尾的路径总得分 = t-1时刻以任意 $y_{t-1}$ 结尾的路径得分 + 转移分数 $T_{y_{t-1}, y_t}$ + 发射分数 $E_{y_t}^{t-1}$：
$$A^t_{y_t}=log(\sum_{y_{t-1}}{e^{A_{y_{t-1}^{t-1}}+T_{y_{t-1}, y_t}+E_{y_t}^t}})$$

记录其中tag的状态为 $y_t=i, y_{t-1}=j$ ，可以得到
$$A^t_i=log(\sum_{j}{e^{A_j^{t-1}+T_{j, i}+E_i^t}})$$

# 3. 步骤拆解
从t=0到t=1过程
|STATE|SOS|B|I|O|EOS|
|----|----|----|----|----|----|
|t=0时刻各个以 $tag_j$ 结尾的路径分数|$A_1^0$|$A_2^0$|$A_3^0$|$A_4^0$|$A_5^0$|
|t=0的 $tag_j$ --> t=1时 $tag_i=1 $的转移分数|$T_{1,1}$|$T_{2,1}$|$T_{3,1}$|$T_{4,1}$|$T_{5,1}$|
|t=1时 $tag_i=1$ 的发射分数|$E_1^1$|$E_1^1$|$E_1^1$|$E_1^1$|$E_1^1$|

t=0的 $tag_j$ --> t=1时 $tag_i=1$ 的各条子路径的得分为
$$[A_1^0+T_{1,1}+E_1^1, A_2^0+T_{2,1}+E_1^1, A_3^0+T_{3,1}+E_1^1, A_4^0+T_{4,1}+E_1^1, A_5^0+T_{5,1}+E_1^1]$$

t=1时刻以 $tag_i=1$ 结尾的路径分数和为各个子路径得分的`logsumexp`
$$A_1^1=log(\sum_j{e^{A_j^0+T_{j,1}+E_1^1}})$$

t=1时刻以 $tag_i$ 结尾的路径分数
$$A_i^1=log(\sum_j{e^{A_j^0+T_{j,i}+E_i^1}})$$

从t=1到t=2过程，依次类推
|STATE|SOS|B|I|O|EOS|
|----|----|----|----|----|----|
|t=1时刻各个以 $tag_j$ 结尾的路径分数|$A_1^1$|$A_2^1$|$A_3^1$|$A_4^1$|$A_5^1$|
|t=1的 $tag_j$ --> t=2时 $tag_i=1$ 的转移分数|$T_{1,1}$|$T_{2,1}$|$T_{3,1}$|$T_{4,1}$|$T_{5,1}$|
|t=2时 $tag_i=1$ 的发射分数|$E_1^2$|$E_1^2$|$E_1^2$|$E_1^2$|$E_1^2$|

t=1的 $tag_j$ --> t=2时 $tag_i=1$ 的各条子路径的得分为
$$[A_1^1+T_{1,1}+E_1^2, A_2^1+T_{2,1}+E_1^2, A_3^1+T_{3,1}+E_1^2, A_4^1+T_{4,1}+E_1^2, A_5^1+T_{5,1}+E_1^2]$$

第一条子路径得分为前序各个子路径的似然概率之和
$$A_1^1+T_{1,1}+E_1^2=log(\sum_j{e^{A_j^0+T_{j,1}+E_1^1}})+T_{1,1}+E_1^2=log(\sum_j{e^{A_j^0+T_{j,1}+E_1^1}}\times e^{T_{1,1}}\times e^{E_1^2})=log(\sum_j{e^{A_j^0+T_{j,1}+E_1^1+T_{1,1}+E_1^2}})$$

t=2时刻以 $tag_i$ 结尾的路径分数
$$A_i^2=log(\sum_j{e^{A_j^1+T_{j,i}+E_i^2}})$$

# 4. 代码
```python
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(1)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        tag_size = feats.size(1)

        # 终点为各个tag的路径的score，初始状态认为第0步状态为START的得分为0，其他的为-10000
        forward_var = torch.full((1, tag_size), -10000.)
        forward_var[0][self.tag_to_ix[START_TAG]] = 0

        # 遍历seq_len
        for i in range(feats.size(0)):
            scores = []
            # 遍历所有的标签
            for next_tag, emit_score in enumerate(feats[i]):

                # 发射分数
                emit_score = emit_score.view(1, -1).expand(1, tag_size)

                # 所有状态转移到next_tag上的概率
                tran_score = self.transitions[next_tag].view(1, -1)

                # 终点为next_tag的路径的score
                score_one_tag = log_sum_exp(forward_var + emit_score + tran_score)
                scores.append(score_one_tag)

            # 终点为各个tag的路径的score
            forward_var = torch.stack(scores).view(1, tag_size)

        # 加上第T步转移到STOP_TAG的转移分数
        final_score = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]

        return log_sum_exp(final_score)

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        scores = torch.zeros(1)

        # 前面加上起始的START_TAG
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        # 遍历seq_len，+ 每一步的转移分数 + 每一步的发射分数
        for i in range(len(feats)):
            scores += self.transitions[tags[i+1], tags[i]] + feats[i][tags[i+1]]

        # 加上到STOP_TAG的转移分数
        scores += self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return scores

    def _viterbi_decode(self, feats):
        '''
        :param feats: [seq_len, tag_size]
        :return:
        '''
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] 保存上一步的tag i + 从tag i转移到当前tag的转移分数
                # 这里不用考虑发射分数，因为这里是求max
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)  # 求以next_tag为终点的路径得分最大的上一步tag i
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))  # 求以next_tag为终点的最大路径得分

            # 加上发射分数emit_score
            # 和_forward_alg一样，存储到当前步tag为i的总得分
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)  # 存储当前步骤得分最高的tag

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)  # 最后一步骤使得路径得分最大的tag i
        path_score = terminal_var[0][best_tag_id]

        # 回溯找到最优路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # 获得lstm层的输出，[seq_len, num_label]
        feats = self._get_lstm_features(sentence)

        # 计算分母，所有路径得分之和
        forward_score = self._forward_alg(feats)

        # 计算golden label的路径得分
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        # 获得lstm的输出 [seq_len, hdsz]
        lstm_feats = self._get_lstm_features(sentence)

        # 维特比解码得到最优路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!
```