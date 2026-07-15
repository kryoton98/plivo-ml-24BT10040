# Starter

Your loop:
  python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
  python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt

Baseline run ~1.5–3 min on a laptop CPU. Log every run in RUNLOG.md.
Everything in train.py and model.py is changeable; the caps and the
evaluate.py interface are not.

Before time is up, your submission folder needs: ckpt.pt, your code,
RUNLOG.md, NOTES.md, and SUMMARY.html (see the assignment brief,
"Deliverables" section).
