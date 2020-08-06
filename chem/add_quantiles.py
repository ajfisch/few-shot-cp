import glob

for quantile in [0.95, 0.90, 0.80, 0.70]:
    for loss in ["quantile", "mse"]:
        setting = "q=%2.2f,l=%s,mol=false" % (quantile, loss)
        ckpt = "../ckpts/chembl/quantiles/random/q=%2.2f,l=%s,mol=false" % (quantile, loss)
        ckpt += "/lightning_logs/version_0/checkpoints/"
        ckpt = glob.glob(ckpt + "epoch*")
        assert(len(ckpt) == 1)
        ckpt = ckpt[0]
        print("python create_cp_data.py " +
              "--ckpt %s " % ckpt +
              "--output_dir ../data/chembl/random/conformal/proto/val/k=10/%s" % setting)
