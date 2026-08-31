# 2026-09-01, tianqi, 4 min English talk aligned with the 6-page demo PPT
# Demo script (4:00, English)

Record **YouTube Public** and paste the link into Devpost. OBS 1080p. Terminal ≥16 pt if you flash `predict.py`.

**Deck:** `docs/slides/TechJam_Challenge5_demo.pptx` (6 slides). Speaker notes on each slide are the same lines as below. Team deep-dive stays on `TechJam_Challenge5.pptx`.

**Do not** build Streamlit. **Do not** say DINOv2 0.90 as the submit score. Submit is CLIP-B last-4 **0.990**; two checkpoints → mean-logit fuse **0.993**. Do not say D3/D4/D5/D6.

Devpost lists a 3-minute video; this talk is **4:00**. If you must cut, drop the mix-in counts on slide 2 (keep “SID is 90%”).

Pace ≈ 125–130 wpm. Total ≈ 510 words.

# end

## Slide 1 — Cover (0:00–0:25)

**On screen:** title, TikTok TechJam 2026, Team Jambuddy, five names.

**Say:**
> Hi, we are Team Jambuddy. After a photo is shared, it is JPEG-compressed, blurred, and cropped. The contest score is point-five times clean AUC plus point-five times the mean AUC under fourteen official transforms — not accuracy at threshold zero-point-five. Our detector is CLIP-B, about eighty-six million parameters, well under the two-billion cap.

## Slide 2 — Mixed data (0:25–1:10)

**On screen:** SID banner (~140k, ~90%) and per-generator mix-in counts.

**Say:**
> Most of the training set is still SID_Set: about one hundred and forty thousand images, ninety percent of train — seventy thousand real photos and seventy thousand FLUX fakes. We do not stack more FLUX. We replace an equal number of SID FLUX with a thin mix-in of other generators: WildFake original SD three thousand nine hundred; SDXL fifteen hundred; PixArt, Stable Diffusion 3.5, Flux.2, nano banana, and GPT-image at fifteen hundred each; ADM and DDPM at one thousand each. These fakes are self-built from local checkpoints and ComfyUI, and we will open-source the mix. No GAN in train: the contest and EvalGEN are diffusion, flow, and autoregressive. Hunyuan is not used.

## Slide 3 — Fuse architecture (1:10–1:55)

**On screen:** CLIP-B last-4 plus mean-logit fuse diagram.

**Say:**
> We unfreeze the last four CLIP-B vision blocks. That scores zero-point-nine-nine-zero on the official four-hundred screen. A larger backbone does not help: CLIP-L last-four is zero-point-nine-eight-zero; ResNet SID is zero-point-seven-seven-nine; DINOv2 is about zero-point-seven-nine. Resize three-three-six, RGB plus frequency, and unfreezing the first four all lose on the contest formula. If two checkpoints are allowed, we average logits of last-four and a mixed-data head. That fuse is zero-point-nine-nine-three. Inference is one command: a folder of images in, JSON out, and pred is the probability the image is AIGC. Official val never enters training.

Optional 10 s cutaway (same clock, do not add a seventh beat):

```bash
python predict.py data/val/fake out.json \
  --ckpt experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt \
  --ckpt-b experiments/clipb16_linear_sid_d3_mix/ckpts/best.pt
```

## Slide 4 — Evaluation (1:55–2:45)

**On screen:** left official-val curves; right EvalGEN / Nova.

**Say:**
> Left is the contest score. Official val is COCO reals plus DALL·E Advanced. Frozen SID on the full thirteen thousand eight hundred forty-three images is zero-point-nine-six-six. Last-four on that full set is zero-point-nine-eight-nine — same ranking as the four-hundred screen. Right is EvalGEN: Flux, GoT, Infinity, OmniGen, and Nova, never trained. Nova is the hard family. Mixed data lifts Nova recall at zero-point-five from zero-point-four-nine to zero-point-eight-six, and Nova AUC to zero-point-nine-eight-eight. Fuse keeps last-four’s DALL·E ranking and the mixed head’s Nova AUC.

## Slide 5 — Bad-case gallery (2:45–3:25)

**On screen:** FN Badcases (left) and FP Badcases (right).

**Say:**
> We ship an HTML gallery: false positives versus false negatives, sorted by worst prediction. On the full official val, frozen SID has one hundred fifty false positives and one thousand eight hundred sixty-five false negatives. Many misses are non-photoreal DALL·E — anime and illustration — scored near zero-point-zero-zero-one. False positives are real COCO photos scored above zero-point-nine-eight. At threshold zero-point-five, fuse on the four-hundred screen is one false positive and forty-four false negatives: low false-accusation, still conservative. A high AUC does not mean zero-point-five is the right operating point.

## Slide 6 — 15 official conditions (3:25–4:00)

**On screen:** fifteen-condition AUROC table.

**Say:**
> The contest formula is half clean AUC and half the mean of fourteen transform keys: JPEG, blur, resize, noise, jitter, and center crop. Fuse stays at or above zero-point-nine-eight-four on every key. The weakest keys are JPEG quality thirty and resize by one quarter. Mixed data dips there; SID is flatter but lower. The repository is public. Install CLIP-B, load the checkpoint, run predict.py. Thank you.

## Before you record

- [ ] Notes pane matches this script; do not ad-lib D-codes or DINOv2 0.90
- [ ] Numbers: last-4 **0.990**, fuse **0.993**, last-4 full val **0.989**
- [ ] No third-party logos on camera
- [ ] YouTube → Public → paste into Devpost
# end
