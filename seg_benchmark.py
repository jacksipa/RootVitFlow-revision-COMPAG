"""
seg_benchmark.py (Prueba3) — Revisor 1 C3/C6, Revisor 3 C3

Re-evaluación de la segmentación (U-Net, SwinUNETR, SAM) con:
  * MISMA partición que la clasificación (splits/assignments.csv, esquema 'holdout', rep 0):
    test = 4 plantas nunca vistas (rb3, rb8, sb13, sb15; 60 imágenes), validación = 2 plantas,
    entrenamiento = 10 plantas. Validación separada del test (en los notebooks originales eran la misma).
  * Resolución completa (3072x4096): entrenamiento por parches con muestreo positivo/negativo 2:1
    (el foreground ocupa ~0.13% de los píxeles) e inferencia por ventana deslizante sobre la imagen completa.
    En los notebooks originales la imagen completa se reducía a 512x512 (U-Net/Swin) o 256x256 (SAM).
  * SAM sin usar la máscara verdadera para el prompt: caja = parche completo (en el notebook original la
    caja se calculaba de la máscara GT, y la validación usaba los logits del último batch de entrenamiento).
  * Mismas métricas para todos, en el test completo y por imagen: Dice, IoU, precisión, recall,
    clDice (Shit et al., 2021) y skeleton-F1 con tolerancia de 2 px.

GT = máscaras binarias de RootPainter (allbin). Uso:
  python seg_benchmark.py --model unet --max_minutes 40
  python seg_benchmark.py --model swinunetr --max_minutes 60
  python seg_benchmark.py --model sam --max_minutes 45
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config_transf as cfg  # noqa: E402

OUT = os.path.join(cfg.RESULTS_DIR, "segmentation")
SAM_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
SAM_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------------------------- datos
def load_split():
    A = pd.read_csv(os.path.join(cfg.SPLITS_DIR, "assignments.csv"))
    A = A[(A.scheme == "holdout") & (A.rep == 0)]
    return {r: sorted(g.fname.str.replace(".png", "", regex=False)) for r, g in A.groupby("role")}


def load_pair(name):
    img = cv2.cvtColor(cv2.imread(os.path.join(cfg.ORIG_IMG_DIR, name + ".jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    msk = (cv2.imread(os.path.join(cfg.IMG_DIR, name + ".png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    assert img.shape[:2] == msk.shape
    return img, msk


class PatchSampler:
    def __init__(self, names, patch, rng, pos_frac=0.67):
        self.data = [load_pair(n) for n in names]
        self.fg = [np.argwhere(m > 0) for _, m in self.data]
        self.patch, self.rng, self.pos_frac = patch, rng, pos_frac

    def sample(self, augment=True):
        i = self.rng.integers(len(self.data))
        img, msk = self.data[i]
        H, W = msk.shape
        p = self.patch
        if self.rng.random() < self.pos_frac and len(self.fg[i]):
            cy, cx = self.fg[i][self.rng.integers(len(self.fg[i]))]
            y0 = int(np.clip(cy - p // 2 + self.rng.integers(-p // 4, p // 4), 0, H - p))
            x0 = int(np.clip(cx - p // 2 + self.rng.integers(-p // 4, p // 4), 0, W - p))
        else:
            y0, x0 = self.rng.integers(0, H - p), self.rng.integers(0, W - p)
        x = img[y0:y0 + p, x0:x0 + p].astype(np.float32) / 255.0
        y = msk[y0:y0 + p, x0:x0 + p].copy()
        if augment:
            if self.rng.random() < .5:
                x, y = x[:, ::-1], y[:, ::-1]
            if self.rng.random() < .5:
                x, y = x[::-1], y[::-1]
            k = self.rng.integers(4)
            x, y = np.rot90(x, k), np.rot90(y, k)
            if self.rng.random() < .3:  # brillo/contraste
                x = np.clip((x - .5) * self.rng.uniform(.75, 1.25) + .5 + self.rng.uniform(-.1, .1), 0, 1)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def batch(self, n, augment=True):
        xs, ys = zip(*[self.sample(augment) for _ in range(n)])
        return np.stack(xs), np.stack(ys)


# ----------------------------------------------------------------------------- métricas
def seg_metrics(pred, gt, tol=2):
    from skimage.morphology import skeletonize
    from scipy.ndimage import distance_transform_edt
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    out = {"dice": 2 * tp / max(2 * tp + fp + fn, 1), "iou": tp / max(tp + fp + fn, 1),
           "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
           "pred_fg_frac": pred.mean(), "gt_fg_frac": gt.mean(), "empty_pred": int(pred.sum() == 0)}
    sp, sg = skeletonize(pred), skeletonize(gt)
    tprec = (sp & gt).sum() / max(sp.sum(), 1)
    tsens = (sg & pred).sum() / max(sg.sum(), 1)
    out["cldice"] = 2 * tprec * tsens / max(tprec + tsens, 1e-12)
    dg = distance_transform_edt(~sg) if sg.any() else np.full(sg.shape, np.inf)
    dp = distance_transform_edt(~sp) if sp.any() else np.full(sp.shape, np.inf)
    sk_p = (dg[sp] <= tol).mean() if sp.any() else 0.0
    sk_r = (dp[sg] <= tol).mean() if sg.any() else 0.0
    out.update({"skel_precision": sk_p, "skel_recall": sk_r,
                "skel_f1": 2 * sk_p * sk_r / max(sk_p + sk_r, 1e-12)})
    return out


# ----------------------------------------------------------------------------- modelos
def build(model_name, device):
    import torch
    if model_name == "unet":
        from monai.networks.nets import UNet
        net = UNet(spatial_dims=2, in_channels=3, out_channels=2, channels=(16, 32, 64, 128, 256),
                   strides=(2, 2, 2, 2), num_res_units=2)
        return net.to(device), dict(patch=512, bs=8, lr=1e-3)
    if model_name == "swinunetr":
        from monai.networks.nets import SwinUNETR
        net = SwinUNETR(in_channels=3, out_channels=2, feature_size=48, spatial_dims=2)
        return net.to(device), dict(patch=512, bs=4, lr=1e-4)
    if model_name == "sam":
        from transformers import SamModel
        net = SamModel.from_pretrained("facebook/sam-vit-base")
        for n, p in net.named_parameters():
            if n.startswith("vision_encoder") or n.startswith("prompt_encoder"):
                p.requires_grad_(False)
        return net.to(device), dict(patch=1024, bs=4, lr=1e-4)
    raise ValueError(model_name)


def forward(model_name, net, x):
    """x: tensor [B,3,P,P] en [0,1]. Devuelve logits de foreground [B,1,P,P]."""
    import torch
    import torch.nn.functional as F
    if model_name in ("unet", "swinunetr"):
        o = net(x)
        return (o[:, 1:2] - o[:, 0:1])
    P = x.shape[-1]
    mean = torch.tensor(SAM_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(SAM_STD, device=x.device).view(1, 3, 1, 1)
    xs = (x - mean) / std
    if P != 1024:
        xs = F.interpolate(xs, size=(1024, 1024), mode="bilinear", align_corners=False)
    with torch.no_grad():
        emb = net.get_image_embeddings(xs)
    boxes = torch.tensor([[[0, 0, 1023, 1023]]], dtype=torch.float32, device=x.device).repeat(x.shape[0], 1, 1)
    o = net(image_embeddings=emb, input_boxes=boxes, multimask_output=False).pred_masks  # [B,1,1,256,256]
    return F.interpolate(o[:, 0], size=(P, P), mode="bilinear", align_corners=False)


def predict_full(model_name, net, img, device, patch):
    import torch
    x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if model_name in ("unet", "swinunetr"):
            from monai.inferers import sliding_window_inference
            o = sliding_window_inference(x, (patch, patch), sw_batch_size=8, overlap=0.25, mode="gaussian",
                                         predictor=lambda t: forward(model_name, net, t))
        else:
            H, W = x.shape[-2:]
            Hp, Wp = int(np.ceil(H / patch) * patch), int(np.ceil(W / patch) * patch)
            xp = torch.nn.functional.pad(x, (0, Wp - W, 0, Hp - H))
            o = torch.zeros((1, 1, Hp, Wp), device=device)
            for y0 in range(0, Hp, patch):
                for x0 in range(0, Wp, patch):
                    o[..., y0:y0 + patch, x0:x0 + patch] = forward(model_name, net, xp[..., y0:y0 + patch, x0:x0 + patch]).float()
            o = o[..., :H, :W]
    return (o[0, 0].float() > 0).cpu().numpy()


# ----------------------------------------------------------------------------- main
def main():
    import torch
    from monai.losses import DiceCELoss
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["unet", "swinunetr", "sam"])
    ap.add_argument("--max_iters", type=int, default=8000)
    ap.add_argument("--max_minutes", type=float, default=45)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bs", type=int, default=None)
    args = ap.parse_args()

    out = os.path.join(OUT, args.model)
    os.makedirs(os.path.join(out, "pred_test"), exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    split = load_split()
    net, hp = build(args.model, device)
    if args.bs:
        hp["bs"] = args.bs
    P = hp["patch"]
    print(f"[{args.model}] train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} "
          f"patch={P} bs={hp['bs']} lr={hp['lr']}", flush=True)

    tr = PatchSampler(split["train"], P, rng)
    va = PatchSampler(split["val"], P, np.random.default_rng(7))
    Xv, Yv = va.batch(48, augment=False)  # parches fijos de validación (2 plantas no vistas)
    del va

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=3e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_iters)
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=False)
    best, best_it, log = -1, 0, []
    t0 = time.time()
    ckpt = os.path.join(out, "best.pt")
    for it in range(1, args.max_iters + 1):
        net.train()
        xb, yb = tr.batch(hp["bs"])
        x = torch.from_numpy(xb).permute(0, 3, 1, 2).to(device)
        y = torch.from_numpy(yb)[:, None].float().to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = forward(args.model, net, x)
        loss = loss_fn(logit.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % args.val_every == 0 or (time.time() - t0) / 60 > args.max_minutes:
            net.eval()
            ds = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vb = min(8, hp["bs"])  # lote de validación acotado por memoria (SAM: parches de 1024)
                for i in range(0, len(Xv), vb):
                    xv = torch.from_numpy(Xv[i:i + vb]).permute(0, 3, 1, 2).to(device)
                    pv = (forward(args.model, net, xv).float() > 0).cpu().numpy()[:, 0]
                    for a, b in zip(pv, Yv[i:i + vb]):
                        tp = (a & (b > 0)).sum()
                        ds.append(2 * tp / max(a.sum() + b.sum(), 1))
            vd = float(np.mean(ds))
            log.append({"iter": it, "loss": float(loss), "val_dice_patches": vd, "minutes": (time.time() - t0) / 60})
            print(f"  it {it} loss {float(loss):.4f} val_dice {vd:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
            if vd > best:
                best, best_it = vd, it
                torch.save({k: v for k, v in net.state_dict().items()}, ckpt)
            pd.DataFrame(log).to_csv(os.path.join(out, "train_log.csv"), index=False)
            if (time.time() - t0) / 60 > args.max_minutes:
                print("  [tiempo máximo alcanzado]", flush=True)
                break
    train_min = (time.time() - t0) / 60
    del tr

    # ---------------- test en resolución completa ----------------
    net.load_state_dict(torch.load(ckpt, map_location=device))
    rows = []
    for n in split["test"]:
        img, gt = load_pair(n)
        pred = predict_full(args.model, net, img, device, P)
        m = seg_metrics(pred, gt)
        m["fname"] = n
        m["plant"] = n.split("_")[0]
        rows.append(m)
        cv2.imwrite(os.path.join(out, "pred_test", n + ".png"), (pred * 255).astype(np.uint8))
        print(f"  test {n}: dice={m['dice']:.3f} cldice={m['cldice']:.3f} skelF1={m['skel_f1']:.3f}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(out, "test_per_image.csv"), index=False)
    cols = ["dice", "iou", "precision", "recall", "cldice", "skel_precision", "skel_recall", "skel_f1", "empty_pred"]
    summ = {"model": args.model, "best_iter": best_it, "best_val_dice_patches": best,
            "train_minutes": round(train_min, 1), "iters_done": log[-1]["iter"] if log else 0,
            "patch": P, "batch": hp["bs"], "lr": hp["lr"],
            "n_params_trainable": int(sum(p.numel() for p in net.parameters() if p.requires_grad)),
            **{f"{c}_mean": float(D[c].mean()) for c in cols}, **{f"{c}_sd": float(D[c].std()) for c in cols}}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
