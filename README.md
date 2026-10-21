# Independent vs. Paired Two-View Training for Corruption-Robust Thyroid Ultrasound Segmentation

Code to reproduce the training, evaluation, statistical analysis, figures and
tables for the paper. Two conditions for presenting two augmented views per
training step, at matched budget and matched total corruption exposure, are
compared:

- **Paired (ρ = -1):** view 1 is always clean, view 2 is always the corrupted
  version of the same image.
- **Independent (ρ = 0):** each view is independently corrupted with
  probability 0.5.

Both conditions supervise both views against the ground-truth mask with a
Dice+BCE loss and no consistency/agreement term. A single-view baseline
(one view per step, corrupted w.p. 0.5, one loss) is also included as a
conventional noise-augmentation reference point.

## Repository layout

```
src/                     model, losses, dataset loaders, transforms, metrics,
                         eval engine, config, corruption functions
train_two_view.py        trains the Paired / Independent conditions
train_singleview.py      trains the single-view baseline
evaluate_zeroshot.py     zero-shot sweep across four corruption families
evaluate_matched.py      train-matched sweep for a given corruption
evaluate_singleview.py   compares the single-view baseline to both conditions
analyze_blur_nonmonotonicity.py   non-monotonicity test on the blur x DDTI result
select_qualitative_cases.py       pre-registered case selection for Fig. "qualitative"
figures/                 one script per figure, shared style in figures/style.py
tables/                  LaTeX table generators (booktabs)
```

## Installation

```
pip install -r requirements.txt
```

Requires a CUDA GPU for training in reasonable time; evaluation and figure
generation also run on CPU.

## Data

This repository does not redistribute the datasets. Obtain them from their
original sources and lay them out as:

```
<data-root>/TN3K/train/images/*.png
<data-root>/TN3K/train/masks/*.png
<data-root>/TN3K/test/images/*.png
<data-root>/TN3K/test/masks/*.png
<ddti-root>/images/*.png
<ddti-root>/masks/*.png
```

TN3K and DDTI are the two thyroid ultrasound datasets used; TN3K is the
training/in-distribution set, DDTI is evaluated zero-shot only.

## Checkpoint layout

Scripts take explicit `--ckpt-dir` / `--*-ckpt-dir` arguments; there is no
hardcoded location. The convention used in the examples below is:

```
experiments/<condition>/<corruption>/seed<seed>/best_model.pth
```

with `<condition>` in `{paired, independent, singleview}`.

## Reproducing the results

1. **Train the speckle-trained checkpoints** (used for the zero-shot sweep
   across all four corruption families), 5 seeds x 2 conditions:

   ```
   for cond in paired independent; do
     for seed in 17 2024 42 123 987; do
       python train_two_view.py --condition $cond --corruption speckle \
         --train-severity 0.15 --seed $seed \
         --data-root <data-root> \
         --ckpt-dir experiments/$cond/speckle/seed$seed
     done
   done
   ```

2. **Train the matched checkpoints** for additive Gaussian noise and Gaussian
   blur (train severities 0.1 and 1.0 respectively), same loop with
   `--corruption gaussian --train-severity 0.1` and
   `--corruption blur --train-severity 1.0`, writing to
   `experiments/$cond/gaussian/seed$seed` and `experiments/$cond/blur/seed$seed`.

3. **Train the single-view baseline** (speckle, 5 seeds):

   ```
   for seed in 17 2024 42 123 987; do
     python train_singleview.py --seed $seed --data-root <data-root> \
       --ckpt-dir experiments/singleview/speckle/seed$seed
   done
   ```

4. **Evaluate zero-shot** (all four corruption families, both datasets):

   ```
   python evaluate_zeroshot.py --data-root <data-root> --ddti-root <ddti-root> \
     --paired-ckpt-dir experiments/paired/speckle \
     --independent-ckpt-dir experiments/independent/speckle \
     --out results/zeroshot_analysis.json
   ```

5. **Evaluate train-matched** (repeat for `gaussian` and `blur`):

   ```
   python evaluate_matched.py --corruption gaussian \
     --data-root <data-root> --ddti-root <ddti-root> \
     --paired-ckpt-dir experiments/paired/gaussian \
     --independent-ckpt-dir experiments/independent/gaussian \
     --out results/gaussian_matched_analysis.json \
     --zero-shot-json results/zeroshot_analysis.json
   ```

6. **Evaluate the single-view baseline**:

   ```
   python evaluate_singleview.py --data-root <data-root> --ddti-root <ddti-root> \
     --singleview-ckpt-dir experiments/singleview/speckle \
     --zero-shot-json results/zeroshot_analysis.json \
     --out results/singleview_analysis.json
   ```

7. **Figures and tables** read directly from the JSON files produced above;
   see each script's `--help` for its exact inputs. Example:

   ```
   python figures/fig_zeroshot_sweep.py --zero-shot-json results/zeroshot_analysis.json
   python tables/gen_tables.py --zero-shot-json results/zeroshot_analysis.json \
     --gaussian-matched-json results/gaussian_matched_analysis.json \
     --blur-matched-json results/blur_matched_analysis.json --out-dir tables/out
   ```

All seeds, severities and the statistical protocol (paired t-test on per-seed
differences, Cohen's d_z, Holm correction within each corruption's severity
family per dataset) match the paper.

## License

MIT. See `LICENSE`.
