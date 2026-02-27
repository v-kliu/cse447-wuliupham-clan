#!/usr/bin/env python
import os
import re
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

"""
CHECKPOINT 1:
dataset: 888 made-up UW vs Oregon sentences (lol)
model: basic n-gram
example accuracy: 80 something?
test accuracy: 

CHECKPOINT 2:
dataset: 10k sentences x 10 languages (Leipzig newspaper, kinda formal/off-domain)
model: n-gram with interpolation
example accuracy: ~80% (misleading, example data is tiny english-only)
test accuracy: 36.2% real TA test, 88s runtime (died on multilingual inputs)

CHECKPOINT 3:
try 1:
dataset: 10k sentences x 10 languages, same newspaper data as cp2
model: character-level transformer + KV cache
example accuracy: 69%
test accuracy: n/a 

try 2:
dataset: 10k sentences x 25 languages, Leipzig across more languages
model: character-level transformer + KV cache
vocab: 5024 -> 6484 chars (now covers hindi, hebrew, korean etc)
example accuracy: 53% (not reliable, example data is english-heavy)
test accuracy: tbd
"""

# model config 
EMBED_DIM   = 128
NUM_HEADS   = 4
NUM_LAYERS  = 4
FFN_DIM     = 512
MAX_SEQ_LEN = 256
DROPOUT     = 0.1

# training config 
BATCH_SIZE  = 128
SEQ_LEN     = 128
EPOCHS      = 3
LR          = 3e-4



class CharVocab:
    """char <-> int mapping built from training text"""

    def __init__(self):
        self.charToIdx = {}
        self.idxToChar = {}
        self.unkIdx    = 0
        self.padIdx    = 1

    def build(self, texts):
        # count chars across all training lines
        counts = {}
        for t in texts:
            for c in t:
                counts[c] = counts.get(c, 0) + 1

        # keep chars that appear at least twice (cut noise)
        specials = ['<unk>', '<pad>']
        allChars = specials + sorted(c for c, cnt in counts.items() if cnt >= 2)

        self.charToIdx = {c: i for i, c in enumerate(allChars)}
        self.idxToChar = {i: c for i, c in enumerate(allChars)}
        self.unkIdx    = self.charToIdx['<unk>']
        self.padIdx    = self.charToIdx['<pad>']

    def encode(self, text):
        return [self.charToIdx.get(c, self.unkIdx) for c in text]

    def __len__(self):
        return len(self.charToIdx)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(
                {'charToIdx': self.charToIdx,
                 'idxToChar': {str(k): v for k, v in self.idxToChar.items()}},
                f, ensure_ascii=False
            )

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        v.charToIdx = d['charToIdx']
        v.idxToChar = {int(k): val for k, val in d['idxToChar'].items()}
        v.unkIdx    = v.charToIdx.get('<unk>', 0)
        v.padIdx    = v.charToIdx.get('<pad>', 1)
        return v


class MultiHeadSelfAttn(nn.Module):
    """
    standard mha + optional kv cache for fast incremental decoding.

    kv cache idea:
      - first pass: process full context, store K and V tensors
      - next token: only compute Q for the new char, concat with cached K/V
      - avoids recomputing attention for chars we already processed
    """

    def __init__(self, embedDim, numHeads, dropout=0.1):
        super().__init__()
        assert embedDim % numHeads == 0, "embedDim must be divisible by numHeads"
        self.numHeads = numHeads
        self.headDim  = embedDim // numHeads
        self.scale    = self.headDim ** -0.5  # 1/sqrt(d_k) to keep softmax stable

        # separate q, k, v projections (easier to read than fused)
        self.Wq = nn.Linear(embedDim, embedDim, bias=False)
        self.Wk = nn.Linear(embedDim, embedDim, bias=False)
        self.Wv = nn.Linear(embedDim, embedDim, bias=False)
        self.Wo = nn.Linear(embedDim, embedDim, bias=False)

        self.attnDrop = nn.Dropout(dropout)

    def _splitHeads(self, x):
        # (batch, seq, dim) -> (batch, heads, seq, headDim)
        b, s, _ = x.shape
        return x.view(b, s, self.numHeads, self.headDim).transpose(1, 2)

    def forward(self, x, mask=None, kvCache=None, useCache=False):
        """
        x:       (batch, seqLen, dim)
        mask:    bool (seqLen, totalLen) - True means block that position
        kvCache: (cachedK, cachedV) or None
                  cachedK/V shape: (batch, heads, pastLen, headDim)
        useCache: whether to return the new (k, v) for next call

        returns (output, newCache)
          output: (batch, seqLen, dim)
          newCache: (k, v) with full history if useCache else None
        """
        b, s, _ = x.shape

        q = self._splitHeads(self.Wq(x))  # (b, heads, s, headDim)
        k = self._splitHeads(self.Wk(x))  # (b, heads, s, headDim)
        v = self._splitHeads(self.Wv(x))  # (b, heads, s, headDim)

        #  kv cache: concat past keys/values with current ones 
        # after this, k and v contain the full history (past + current tokens)
        # q is still just the current tokens - that's the whole point
        if kvCache is not None:
            cachedK, cachedV = kvCache
            k = torch.cat([cachedK, k], dim=2)  # dim=2 is the sequence dim
            v = torch.cat([cachedV, v], dim=2)
        # k shape is now (b, heads, pastLen+s, headDim)
        # v shape is now (b, heads, pastLen+s, headDim)
        # q shape is still (b, heads, s, headDim)

        newCache = (k.detach(), v.detach()) if useCache else None

        # scores: q @ k^T, shape (b, heads, s, pastLen+s)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask: (seqLen, totalLen) -> broadcast over (b, heads, seqLen, totalLen)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        weights = self.attnDrop(F.softmax(scores, dim=-1))

        # weighted sum of values: (b, heads, s, headDim) -> (b, s, dim)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.Wo(out)

        return out, newCache


class TransformerBlock(nn.Module):
    """one decoder block: layernorm -> attn -> residual -> layernorm -> ffn -> residual"""

    def __init__(self, embedDim, numHeads, ffnDim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedDim)
        self.norm2 = nn.LayerNorm(embedDim)
        self.attn  = MultiHeadSelfAttn(embedDim, numHeads, dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(embedDim, ffnDim),
            nn.GELU(),                   # gelu > relu for transformers
            nn.Linear(ffnDim, embedDim),
            nn.Dropout(dropout),
        )
        self.resDrop = nn.Dropout(dropout)

    def forward(self, x, mask=None, kvCache=None, useCache=False):
        # pre-norm attention sub-layer with residual
        attnOut, newCache = self.attn(
            self.norm1(x), mask=mask, kvCache=kvCache, useCache=useCache
        )
        x = x + self.resDrop(attnOut)
        # pre-norm ffn sub-layer with residual
        x = x + self.ffn(self.norm2(x))
        return x, newCache

# hereee

class CharTransformer(nn.Module):
    """
    decoder-only char transformer
    4 layers, 128 dim, 4 heads, ffn=512
    predicts next char given a sequence of chars
    """

    def __init__(self, vocabSize,
                 embedDim=EMBED_DIM, numHeads=NUM_HEADS, numLayers=NUM_LAYERS,
                 ffnDim=FFN_DIM, maxSeqLen=MAX_SEQ_LEN, dropout=DROPOUT):
        super().__init__()
        self.maxSeqLen = maxSeqLen
        self.numLayers = numLayers

        self.charEmb = nn.Embedding(vocabSize, embedDim)
        self.posEmb  = nn.Embedding(maxSeqLen, embedDim)  # learned positional
        self.embDrop = nn.Dropout(dropout)

        self.blocks    = nn.ModuleList([
            TransformerBlock(embedDim, numHeads, ffnDim, dropout)
            for _ in range(numLayers)
        ])
        self.normFinal = nn.LayerNorm(embedDim)
        self.lmHead    = nn.Linear(embedDim, vocabSize, bias=False)

    def _causalMask(self, seqLen, device, pastLen=0):
        """
        causal mask so position i can't peek at future positions.

        returns bool tensor (seqLen, pastLen+seqLen)
          True  = block this (query, key) pair
          False = allow attention

        the math:
          query at local pos i has absolute pos (pastLen + i)
          it can attend to all absolute positions 0 .. pastLen+i
          so mask[i, j] = True when j > pastLen + i

        when pastLen > 0 (cache exists):
          new tokens can freely attend to entire cached past
          e.g. seqLen=1, pastLen=P: mask[0, j] = (j > P)
          since j only goes 0..P, nothing gets masked -> attends to full cache
        """
        totalLen = pastLen + seqLen
        row = torch.arange(seqLen,   device=device).unsqueeze(1)  # (seqLen, 1)
        col = torch.arange(totalLen, device=device).unsqueeze(0)  # (1, totalLen)
        # True where we should NOT attend (future positions)
        return col > (pastLen + row)  # (seqLen, totalLen)

    def forward(self, x, kvCaches=None, useCache=False):
        """
        x:        (batch, seqLen) token indices
        kvCaches: list of (cachedK, cachedV) per layer, or None for fresh run
        useCache: if True, return updated caches for incremental decoding

        returns (logits, newCaches)
          logits:    (batch, seqLen, vocabSize)
          newCaches: list of (k, v) per layer if useCache, else None
        """
        b, s   = x.shape
        device = x.device

        # figure out how many tokens are already in the cache
        pastLen = 0
        if kvCaches is not None and kvCaches[0] is not None:
            # cachedK shape: (batch, heads, pastLen, headDim)
            pastLen = kvCaches[0][0].shape[2]

        # positions start after the cached prefix (so pos embeddings are correct)
        positions = torch.arange(pastLen, pastLen + s, device=device)
        positions = positions.clamp(max=self.maxSeqLen - 1).unsqueeze(0)  # (1, s)

        h = self.embDrop(self.charEmb(x) + self.posEmb(positions))

        # causal mask verification 
        # this mask is the key thing that prevents "cheating" (looking at future chars)
        # during training: (seqLen, seqLen) lower-triangular style
        # during inference with cache: (1, pastLen+1) - all False since new token
        #   attends to full history
        mask = self._causalMask(s, device, pastLen)

        newCaches = []
        for i, block in enumerate(self.blocks):
            layerCache = kvCaches[i] if kvCaches is not None else None
            h, layerNewCache = block(h, mask=mask, kvCache=layerCache, useCache=useCache)
            newCaches.append(layerNewCache)

        logits = self.lmHead(self.normFinal(h))  # (b, s, vocabSize)

        return logits, (newCaches if useCache else None)

class MyModel:
    """wraps CharTransformer to match the train/test CLI from the original code"""

    def __init__(self):
        self.vocab  = None
        self.model  = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def preprocessLine(cls, line):
        # strip the "123\t" prefix that Leipzig corpus adds to each line
        return re.sub(r'^\d+\t', '', line).strip()

    @classmethod
    def load_training_data(cls):
        allData = []
        dataDir = './src/training_data'
        if os.path.isdir(dataDir):
            for fname in sorted(os.listdir(dataDir)):
                if fname.endswith('.txt'):
                    print(f'  loading {fname}...')
                    with open(os.path.join(dataDir, fname), encoding='utf-8') as f:
                        for line in f:
                            clean = cls.preprocessLine(line)
                            if clean:
                                allData.append(clean)
        print(f'  total: {len(allData)} sentences')
        return allData

    @classmethod
    def load_test_data(cls, fname):
        with open(fname, encoding='utf-8') as f:
            return [line[:-1] for line in f]  # drop trailing newline

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write(f'{p}\n')

    def run_train(self, data, workDir):
        # build vocab from all training text
        self.vocab = CharVocab()
        self.vocab.build(data)
        vocabSize  = len(self.vocab)
        print(f'  vocab size: {vocabSize}')

        self.model = CharTransformer(vocabSize=vocabSize).to(self.device)
        numParams  = sum(p.numel() for p in self.model.parameters())
        print(f'  model params: {numParams:,}')
        print(f'  device: {self.device}')

        optim = torch.optim.Adam(self.model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

        # flatten all sentences into one big token buffer (space as sentence sep)
        print('  tokenizing...')
        allToks = []
        spaceIdx = self.vocab.charToIdx.get(' ', self.vocab.unkIdx)
        for line in data:
            allToks.extend(self.vocab.encode(line))
            allToks.append(spaceIdx)
        allToks = torch.tensor(allToks, dtype=torch.long)
        print(f'  total tokens: {len(allToks):,}')

        self.model.train()
        for ep in range(EPOCHS):
            # non-overlapping windows across the token stream
            startIdxs = list(range(0, len(allToks) - SEQ_LEN - 1, SEQ_LEN))
            random.shuffle(startIdxs)

            totalLoss  = 0.0
            numBatches = 0

            for bStart in range(0, len(startIdxs), BATCH_SIZE):
                batchIdxs = startIdxs[bStart: bStart + BATCH_SIZE]
                if not batchIdxs:
                    continue

                # build (batchSize, SEQ_LEN+1) tensor
                seqs   = torch.stack([allToks[i: i + SEQ_LEN + 1] for i in batchIdxs]).to(self.device)
                inpSeq = seqs[:, :-1]   # (batch, SEQ_LEN) - input tokens
                tgtSeq = seqs[:, 1:]    # (batch, SEQ_LEN) - targets (shifted by 1)

                optim.zero_grad()

                # training forward pass 
                # no kvCaches here - full sequence, causal mask handles ordering
                # logits[b, i, :] = distribution over next char after position i
                logits, _ = self.model(inpSeq)  # (batch, SEQ_LEN, vocabSize)

                loss = F.cross_entropy(
                    logits.reshape(-1, vocabSize),
                    tgtSeq.reshape(-1),
                    ignore_index=self.vocab.padIdx
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optim.step()

                totalLoss  += loss.item()
                numBatches += 1

            sched.step()
            avgLoss = totalLoss / max(numBatches, 1)
            print(f'  epoch {ep + 1}/{EPOCHS}  loss={avgLoss:.4f}')

        # sanity check top-3 on the provided example data
        self._validateTop3()

    def _validateTop3(self):
        """local top-3 accuracy check on example/input.txt + example/answer.txt"""
        inPath  = 'example/input.txt'
        ansPath = 'example/answer.txt'
        if not (os.path.exists(inPath) and os.path.exists(ansPath)):
            print('  (no example data found for validation, skipping)')
            return

        testData = self.load_test_data(inPath)
        preds    = self.run_pred(testData)

        with open(ansPath, encoding='utf-8') as f:
            answers = [line.strip() for line in f]

        # top-3 check: correct if the true next char appears anywhere in our 3 guesses
        correct = sum(
            1 for pred, ans in zip(preds, answers)
            if ans.lower() in pred.lower()
        )
        total = len(answers)
        print(f'  top-3 accuracy on example data: {correct}/{total} = {correct / total:.2%}')

    def run_pred(self, data):
        """
        predict top-3 next chars for each prompt using kv cache inference.

        kv cache flow:
          1. feed full context -> model builds and returns K/V for every layer
          2. logits at the last position = prediction for the next char
          3. if we wanted to continue generating, we'd feed just the next token
             with kvCaches=kvCaches, and only compute attention for that 1 token
             (avoiding recomputing attention over the whole context again)
        """
        self.model.eval()
        preds    = []
        fallback = 'e atonirsl'  # common chars to pad if top-k gives us < 3

        with torch.no_grad():
            for prompt in data:
                try:
                    if not prompt:
                        preds.append('e a')
                        continue

                    # cap context to maxSeqLen, take the most recent chars
                    if len(prompt) > self.model.maxSeqLen:
                        prompt = prompt[-self.model.maxSeqLen:]

                    tokens = self.vocab.encode(prompt)
                    inp    = torch.tensor([tokens], dtype=torch.long, device=self.device)

                    #step 1: full context forward, build kv cache 
                    # logits shape: (1, seqLen, vocabSize)
                    # kvCaches[i] = (k, v) for layer i
                    #   k shape: (1, numHeads, seqLen, headDim)
                    #   v shape: (1, numHeads, seqLen, headDim)
                    logits, kvCaches = self.model(inp, useCache=True)

                    #  step 2: grab prediction at the last position 
                    lastLogits = logits[0, -1, :]  # (vocabSize,)

                    #  kv cache shape verification 
                    # assert that cache tensors have the right dims
                    # cachedK: (batch=1, heads=4, seqLen, headDim=32)
                    # cachedV: same shape
                    # when we feed the next token, seqLen becomes seqLen+1
                    assert kvCaches is not None
                    for layerK, layerV in kvCaches:
                        assert layerK.shape == layerV.shape
                        #  (1, NUM_HEADS, len(tokens), EMBED_DIM // NUM_HEADS)
                        assert layerK.shape[1] == NUM_HEADS
                        assert layerK.shape[2] == len(tokens)
                        assert layerK.shape[3] == EMBED_DIM // NUM_HEADS

                    # pick top-3 chars (skip special tokens)
                    topK = torch.topk(lastLogits, k=min(20, len(self.vocab))).indices.tolist()
                    pred = ''
                    for idx in topK:
                        ch = self.vocab.idxToChar.get(idx, '')
                        if ch and ch not in ('<unk>', '<pad>') and ch not in pred:
                            pred += ch
                        if len(pred) >= 3:
                            break

                    # pad with common chars if we didn't hit 3
                    for fb in fallback:
                        if len(pred) >= 3:
                            break
                        if fb not in pred:
                            pred += fb

                    preds.append(pred[:3])

                except Exception:
                    preds.append('e a')

        return preds

    def save(self, workDir):
        # save vocab separately so we can rebuild the model on load
        self.vocab.save(os.path.join(workDir, 'vocab.json'))

        torch.save({
            'modelState': self.model.state_dict(),
            'config': {
                'vocabSize': len(self.vocab),
                'embedDim':  EMBED_DIM,
                'numHeads':  NUM_HEADS,
                'numLayers': NUM_LAYERS,
                'ffnDim':    FFN_DIM,
                'maxSeqLen': MAX_SEQ_LEN,
                'dropout':   DROPOUT,
            }
        }, os.path.join(workDir, 'model.checkpoint'))

        print(f'  saved vocab.json + model.checkpoint to {workDir}/')

    @classmethod
    def load(cls, workDir):
        m       = cls()
        m.vocab = CharVocab.load(os.path.join(workDir, 'vocab.json'))

        # weights_only=False 
        ckpt = torch.load(
            os.path.join(workDir, 'model.checkpoint'),
            map_location=m.device,
            weights_only=False
        )
        cfg = ckpt['config']

        m.model = CharTransformer(
            vocabSize=cfg['vocabSize'],
            embedDim=cfg['embedDim'],
            numHeads=cfg['numHeads'],
            numLayers=cfg['numLayers'],
            ffnDim=cfg['ffnDim'],
            maxSeqLen=cfg['maxSeqLen'],
            dropout=cfg.get('dropout', 0.1),
        ).to(m.device)

        m.model.load_state_dict(ckpt['modelState'])
        m.model.eval()
        return m


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir',    help='where to save/load checkpoints', default='work')
    parser.add_argument('--test_data',   help='path to test input file',        default='example/input.txt')
    parser.add_argument('--test_output', help='path to write predictions',      default='pred.txt')
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print(f'Making working directory {args.work_dir}')
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        trainData = MyModel.load_training_data()
        print('Training')
        model.run_train(trainData, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)

    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print(f'Loading test data from {args.test_data}')
        testData = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(testData)
        print(f'Writing predictions to {args.test_output}')
        assert len(pred) == len(testData), f'Expected {len(testData)} preds, got {len(pred)}'
        MyModel.write_pred(pred, args.test_output)

