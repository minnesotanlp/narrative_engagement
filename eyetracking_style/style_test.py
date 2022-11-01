from styleformer import Styleformer
from bleu import list_bleu

with open('/Users/karin/nlp/GYAFC/test.bi.src') as f:
     best_score = 0
     worst_score = 100
     inf2for = Styleformer(0)
     lines = f.readlines()
     for i in range(len(lines) - 1):
         inf = lines[i].split('>')[-1]
         frm = lines[i + 1].split('>')[-1]
         hyp = inf2for.transfer(inf)
         bleu = list_bleu([frm.strip()], [hyp.strip()])
         if bleu > best_score:
             best_score = bleu
             print('new best:', bleu)
             print(inf, frm, hyp)
             print()
         if bleu < worst_score:
             worst_score = bleu
             print('new worst:', bleu)
             print(inf, frm, hyp)
             print()
